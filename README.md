Utilizing a relatively innovative approach to identify general patterns within intricate systems, we employ the analysis presented in this paper: https://arxiv.org/pdf/1809.07100.pdf to the cryptocurrency market.

Cryptocurrencies, an emerging asset class with a market capitalization exceeding $1 trillion, exhibit a notable level of correlation in their price movements. It begs the question: Can we develop a probabilistic prediction for tomorrow's events based on the interplay of cryptocurrency prices today?

To address this query, we employ a novel and systematic methodology for detecting common correlation patterns within financial markets. Our dataset encompasses four years of data from Yahoo Finance, encompassing 146 different cryptocurrencies ranked among the top 500 by market capitalization. We derive various short-term (20-day) correlation frames from the log-return data of these assets, aiming to mitigate non-stationarity. However, this choice introduces inherent noise into our time series, which we counteract by employing the "power mapping" method (refer to https://www.tandfonline.com/doi/abs/10.1080/14697680902748498 for a comprehensive review).

Once we obtain our noise-reduced correlation frames, we proceed to compute a similarity matrix. Employing multidimensional scaling of this matrix, we enhance our ability to perform k-means clustering. Each column of the similarity matrix corresponds to one of our correlation frames. By optimizing k, we determine the number of generic correlation patterns (market states) in the cryptocurrency market, which turns out to be k=4. While two of these patterns merely indicate an increase in market capitalization over the past four years, the remaining two states, referred to as the "crash" and "rally" states, consistently lead to either losses or gains, respectively. On average, the crash state exhibits a daily loss of 0.74%, whereas the rally state experiences an average daily gain of 1.05%. We also assign empirical probabilities to transitions between the four market states, which remain resilient to model drift due to the practically Markovian nature of the cryptocurrency market.






