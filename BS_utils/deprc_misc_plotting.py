maturities = ctx.get('T').unique()

ind = 6

callData = ctx.filter(
    (ctx.col('T') == maturities[ind]) &
    (ctx.col('option_type') == 'C')
)

putData = ctx.filter(
    (ctx.col('T') == maturities[ind]) &
	(ctx.col('option_type') == 'P')
)

plt.figure(figsize=(20, 10))

ask_IV_type = 'ask_IV_union'
bid_IV_type = 'bid_IV_union'
mid_IV_type = 'mid_IV_union'


plt.fill_between(putData.get('m'), putData.get(ask_IV_type), putData.get(bid_IV_type) , alpha=0.5, label='Bid-Ask IV Spread for Put Options')
plt.fill_between(callData.get('m'), callData.get(ask_IV_type), callData.get(bid_IV_type), alpha=0.5, label='Bid-Ask IV Spread for Call Options', color='tab:orange')

plt.plot(putData.get('m'), putData.get(mid_IV_type), label='Put Mid-Price IV', color='b')
plt.plot(callData.get('m'), callData.get(mid_IV_type), label='Call Mid-Price IV', color='r')

plt.plot(putData.get('m'), putData.get('iv_source'), label='Put Yahoo IV', color='b', linestyle='--')
plt.plot(callData.get('m'), callData.get('iv_source'), label='Call Yahoo IV', color='r', linestyle='--')

plt.axvline(x=0, color='black', linestyle='--', label='Spot Price')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title( 'Implied Volatility vs Strike Price for '+ putData.get('expiration').to_list()[0].strftime('%Y-%m-%d') + ' Expiry at ' + dt(2025,4,1).strftime('%Y-%m-%d') )
plt.xlim(-0.5,0.5)
plt.ylim(0,0.5)
plt.legend()
plt.show()

plt.figure(figsize=(20, 10))

plt.fill_between(putData.get('m'), (putData.get(ask_IV_type)-putData.get(mid_IV_type)), (putData.get(bid_IV_type)-putData.get(mid_IV_type)) , alpha=0.5, label='Bid-Ask IV Spread for Put Options')
plt.fill_between(callData.get('m'), (callData.get(ask_IV_type)-callData.get(mid_IV_type)), (callData.get(bid_IV_type)-callData.get(mid_IV_type)), alpha=0.5, label='Bid-Ask IV Spread for Call Options', color='tab:orange')

plt.plot(putData.get('m'), (putData.get('iv_source')-putData.get(mid_IV_type)), label='Put: (CBOE IV - Mid IV)/Mid IV', color='b', linestyle='--')
plt.plot(callData.get('m'), (callData.get('iv_source')-callData.get(mid_IV_type)), label='Call: (CBOE IV - Mid IV)/Mid IV', color='r', linestyle='--')

plt.axvline(x=0, color='black', linestyle='--', label='Mid Price IV')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlim(left=-0.45)
plt.xlim(right=0.45)
plt.ylim(-0.1,0.1)
plt.xlabel('Strike Price')
plt.ylabel('Difference of Implied Volatility')
plt.title( 'Implied Volatility vs Strike Price for '+ putData.get('expiration').to_list()[0].strftime('%Y-%m-%d') + ' Expiry at ' + dt(2025,4,1).strftime('%Y-%m-%d') )
plt.legend()