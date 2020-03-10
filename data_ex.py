import pandas as pd

clickbait = pd.read_csv("./clickbait_data", sep="\n")
non_clickbait = pd.read_csv("./non_clickbait_data", sep="\n")

#add labels
clickbait['label'] = "1"
non_clickbait['label'] = "-1"

#random sample
click_rand = clickbait.sample(n=10000)
non_click_rand = non_clickbait.sample(n=10000)

#concatenate
click_rand.to_csv("random_clickbait.csv")
non_click_rand.to_csv("random_non_clickbait.csv")
#data = pd.concat([click_rand, non_click_rand])
##data.to_csv("merged.csv")