# journal-analysis-project
*Analyzes 4.5+ years of journal entries using NLP in python to derive unique, data-driven insights about my life experience.*

In this project, I analyzed the daily journals that I’ve been writing in google docs for the past four and a half years.

I used WordCloud to visualize my most common words for each year, giving me a brief refresher on my history. The words matched closely with my memory of those years. I then analyzed the polarity of my writing over time using TextBlob, and saw that 2020 was almost twice as positive as any other year. I was surprised how strongly the sentiment corresponded to my feelings; 2020 has been the happiest year of my life by far. Finally, I looked at the length of my journal entries over time, and visualizing length against polarity revealed some interesting insights.

The last thing I did was train a deep learning neural network on my journal text using fast.ai. I had the neural network generate sample journal entries based on various starting words, and it produced some eerily similar entries to how I write.

I wrote an article about the process and findings for this project; you can find it [here](https://towardsdatascience.com/exploring-4-5-years-of-journal-entries-with-nlp-589de6130c2d)
