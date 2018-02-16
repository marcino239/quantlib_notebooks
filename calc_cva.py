# based on works of 2015 Matthias Groncki
# https://github.com/mgroncki/IPythonScripts
#

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#- Redistributions of source code must retain the above copyright notice,
#this list of conditions and the following disclaimer.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#This disclaimer is taken from the QuantLib license
#

import numpy as np
import matplotlib.pyplot as plt
import QuantLib as ql

def calc_cva( hazard_rate=0.02, market_rate=0.03, ir_vol=0.0075, swap_term=5, notional=10000, N=2000, seed=1 ):
    # Setting evaluation date
    today = ql.Date(15,2,2018)
    ql.Settings.instance().setEvaluationDate(today)
    
    # Setup Marketdata
    rate = ql.SimpleQuote( market_rate )
    rate_handle = ql.QuoteHandle(rate)
    dc = ql.Actual365Fixed()
    yts = ql.FlatForward(today, rate_handle, dc)
    yts.enableExtrapolation()
    hyts = ql.RelinkableYieldTermStructureHandle(yts)
    t0_curve = ql.YieldTermStructureHandle(yts)
    euribor6m = ql.Euribor6M(hyts)
    
    # IR vol
    volas = [ql.QuoteHandle(ql.SimpleQuote(ir_vol)), ql.QuoteHandle(ql.SimpleQuote(ir_vol))]
    meanRev = [ql.QuoteHandle(ql.SimpleQuote(0.02))]
    model = ql.Gsr(t0_curve, [today+100], volas, meanRev, 16.)    
    
    # Setup a dummy portfolio with single swap 
    def makeSwap(start, maturity, nominal, fixedRate, index, typ=ql.VanillaSwap.Payer):
        """
        creates a plain vanilla swap with fixedLegTenor 1Y

        parameter:

            start (ql.Date) : Start Date

            maturity (ql.Period) : SwapTenor

            nominal (float) : Nominal

            fixedRate (float) : rate paid on fixed leg

            index (ql.IborIndex) : Index

        return: tuple(ql.Swap, list<Dates>) Swap and all fixing dates


        """
        end = ql.TARGET().advance(start, maturity)
        fixedLegTenor = ql.Period("1y")
        fixedLegBDC = ql.ModifiedFollowing
        fixedLegDC = ql.Thirty360(ql.Thirty360.BondBasis)
        spread = 0.0
        fixedSchedule = ql.Schedule(start,
                                    end, 
                                    fixedLegTenor, 
                                    index.fixingCalendar(), 
                                    fixedLegBDC,
                                    fixedLegBDC, 
                                    ql.DateGeneration.Backward,
                                    False)
        floatSchedule = ql.Schedule(start,
                                    end,
                                    index.tenor(),
                                    index.fixingCalendar(),
                                    index.businessDayConvention(),
                                    index.businessDayConvention(),
                                    ql.DateGeneration.Backward,
                                    False)
        swap = ql.VanillaSwap(typ, 
                              nominal,
                              fixedSchedule,
                              fixedRate,
                              fixedLegDC,
                              floatSchedule,
                              index,
                              spread,
                              index.dayCounter())
        return swap, [index.fixingDate(x) for x in floatSchedule][:-1]

    portfolio = [makeSwap(today + ql.Period("2d"),
                          ql.Period( swap_term, ql.Years ),
                          notional,
                          0.03,
                          euribor6m),
                ]
    
    # Setup pricing engine and calculate the npv
    engine = ql.DiscountingSwapEngine(hyts)
    for deal, fixingDates in portfolio:
        deal.setPricingEngine(engine)
        deal.NPV()
        
    process = model.stateProcess()
    
    # Define evaluation grid
    date_grid = [today + ql.Period(i,ql.Months) for i in range(0,12*6)]
    for deal in portfolio:
        date_grid += deal[1]

    date_grid = np.unique(np.sort(date_grid))
    time_grid = np.vectorize(lambda x: ql.ActualActual().yearFraction(today, x))(date_grid)
    dt = time_grid[1:] - time_grid[:-1]
    
    # Random number generator
    urng = ql.MersenneTwisterUniformRng(seed)
    usrg = ql.MersenneTwisterUniformRsg(len(time_grid)-1,urng)
    generator = ql.InvCumulativeMersenneTwisterGaussianRsg(usrg)
    
    x = np.zeros((N, len(time_grid)))
    y = np.zeros((N, len(time_grid)))
    pillars = np.array([0.0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    zero_bonds = np.zeros((N, len(time_grid), 12))

    for j in range(12):
        zero_bonds[:, 0, j] = model.zerobond(pillars[j],
                                             0,
                                             0)
    for n in range(0,N):
        dWs = generator.nextSequence().value()
        for i in range(1, len(time_grid)):
            t0 = time_grid[i-1]
            t1 = time_grid[i]
            x[n,i] = process.expectation(t0, 
                                         x[n,i-1], 
                                         dt[i-1]) + dWs[i-1] * process.stdDeviation(t0,
                                                  x[n,i-1],
                                                  dt[i-1])
            y[n,i] = (x[n,i] - process.expectation(0,0,t1)) / process.stdDeviation(0,0,t1)
            for j in range(12):
                zero_bonds[n, i, j] = model.zerobond(t1+pillars[j],
                                                     t1,
                                                     y[n, i])

    discount_factors = np.vectorize(t0_curve.discount)(time_grid)
    
    npv_cube = np.zeros((N,len(date_grid), len(portfolio)))
    for p in range(0,N):
        for t in range(0, len(date_grid)):
            date = date_grid[t]
            ql.Settings.instance().setEvaluationDate(date)
            ycDates = [date, 
                       date + ql.Period(6, ql.Months)] 
            ycDates += [date + ql.Period(i,ql.Years) for i in range(1,11)]
            yc = ql.DiscountCurve(ycDates, 
                                  zero_bonds[p, t, :], 
                                  ql.Actual365Fixed())
            yc.enableExtrapolation()
            hyts.linkTo(yc)
            if euribor6m.isValidFixingDate(date):
                fixing = euribor6m.fixing(date)
                euribor6m.addFixing(date, fixing)
            for i in range(len(portfolio)):
                npv_cube[p, t, i] = portfolio[i][0].NPV()
        ql.IndexManager.instance().clearHistories()
    ql.Settings.instance().setEvaluationDate(today)
    hyts.linkTo(yts)
    
    # Calculate the discounted npvs
    discounted_cube = np.zeros(npv_cube.shape)
    for i in range(npv_cube.shape[2]):
        discounted_cube[:,:,i] = npv_cube[:,:,i] * discount_factors
        
    # Calculate the portfolio npv by netting all NPV
    portfolio_npv = np.sum(npv_cube,axis=2)
    discounted_npv = np.sum(discounted_cube, axis=2)
    
    # Setup Default Curve 
    pd_dates =  [today + ql.Period(i, ql.Years) for i in range(11)]
    hzrates = [ hazard_rate * i for i in range(11) ]
    pd_curve = ql.HazardRateCurve(pd_dates,hzrates,ql.Actual365Fixed())
    pd_curve.enableExtrapolation()
    
    # Calculation of the default probs
    defaultProb_vec = np.vectorize(pd_curve.defaultProbability)
    dPD = defaultProb_vec(time_grid[:-1], time_grid[1:])

    # calculate expected exposure
    dE = discounted_npv.copy()
    dE[dE<0] = 0
    dEE = np.sum(dE, axis=0)/N
    dEEstd = np.std( dE / N, axis=0 )
    
    # Calculation of the CVA
    recovery = 0.4
    CVA = (1-recovery) * np.sum(dEE[1:] * dPD)
    return CVA, dEE, dEEstd

def worker_calc_cva( a ):
    # return only CVA
    return calc_cva( **a )[0]
