from sklearn.datasets import fetch_20newsgroups
import io,sys

#文字コード
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


#カテゴリ
categories = ['rec.sport.baseball', 'rec.sport.hockey', 'comp.sys.mac.hardware', 'comp.windows.x']
#学習、テストデータ
twenty_train = fetch_20newsgroups(subset="train", categories=categories,shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset="test", categories=categories, shuffle=True, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer(lowercase=True, stop_words="english", max_df=0.1, min_df= 5).fit(twenty_train.data)
X_train = tfidf_vec.transform(twenty_train.data)
X_test = tfidf_vec.transform(twenty_test.data)
X = tfidf_vec.transform(twenty_train.data)

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=2,max_iter=100,learning_method="batch", random_state=0, n_jobs=-1)

#モデルの生成
lda.fit(X)
lda_X = lda.transform(X)

import mglearn
import numpy as np

#各トピックの重要度SORT
sorting = np.argsort(lda.components_,axis=1)[:, ::-1]
feature_names = np.array(tfidf_vec.get_feature_names())

#分類テスト MS, GTA(Xbox),Xiaomi device Football, 
test = [
    "Microsoft's Groove Music service bowed out months ago, but all the apps have been available for listening to your personal files... until now. The company has revealed that its Android and iOS apps will disappear from their respective online stores on June 1st, 2018, and the apps themselves will be retired (that is, stop working) on December 1st. After that, you'll have to use OneDrive or migrate your songs elsewhere to keep listening in the cloud.The Windows 10 apps for PCs, Windows-powered phones and Xbox One will continue to work for your personal collection, so this isn't a complete shutdown of the software. However, there's little doubt that Microsoft is reducing Groove Music's support to the bare essentials. Now that it has effectively conceded to the likes of Spotify and Apple Music, there's not much point to maintaining apps across platform.",
    "If your Xbox copy of Grand Theft Auto: San Andreas is stashed away on a shelf somewhere, you might want to pull it out as you'll have some use for it again next week. Starting next Thursday, Rockstar Games is adding backwards compatibility for the game on Xbox One, as well as Midnight Club: Los Angeles and the slightly off-brand Rockstar Games presents Table Tennis.The San Andreas compatibility is a boon for GTA fans. The story of Carl Johnson trying to save his family is often hailed as one of the best games ever, and it's one of the highest-selling games of all time. Either the Xbox and Xbox 360 copy will work on your new-fangled console, but Xbox saves won't transfer. If you do have the OG Xbox version, you can download the Xbox 360 HD remaster with better draw distances and the all-important Achievements.These three games join Red Dead Redemption on Rockstar's backwards compatibility roster. As ever, if you purchased digital versions of the games, they'll pop up in your Ready to Install tab when they become available on Xbox One. If you have a disc, you can insert it and download a port of the game. We're looking forward to seeing lots of wheelie record attempts in Los Santos on Twitch very soon.",
    "Out of the handful of new devices from Xiaomi today, it was the Mi 8 Explorer Edition that stole the show with its transparent design. Right after the keynote, I had to fight a crowd to get up close and personal with one of the few demo devices available, and it was worth the effort. While Xiaomi's website has stated that what's on display doesn't correspond to every single component, a company rep claimed that they are actually all legit, especially the Snapdragon chipset -- which is basically free promotion for Qualcomm-- and its surrounding electrical components. Assuming this is indeed the truth, what remains a mystery is whether this kind of layout would suffer from weaker heat dissipation.But for now, I'm all for just gazing at this mesmerizing device. Transparent design is nothing new in the smartphone world, of course. Most recently, we've seen HTC offering translucent options for its U12+ and U11+, then a little while back there was also the transparent Firefox OS phone from Japan's KDDI, but they actually don't show much -- it's usually just the battery, NFC coil, screws and a few ribbons. Xiaomi went the extra mile with its Mi 8 Explorer Edition by showing alleged real parts on the logic board, and such fine details add a lot to this look.Having said that, there's probably too much detail as well. I'm not a fan of those random words dotted around the body, especially the two lines that read like they could have been pulled out of fortune cookies. Seriously, one of them reads: Always believe that something wonderful is about to happen. That's deep. If Xiaomi must keep one line, though, then I'll make do with the innovation for everyone tag on the NFC module.Cheesy lines aside, the Mi 8 Explorer Edition appears to be a fun device to have, and 3,699 yuan (about $580) isn't too bad for a flagship device with Snapdragon 845, 8GB of RAM, 128GB of storage, dual cameras and 3D face scanner. Alas, there's no word on Western availability; there isn't even a date yet for China, so we'll just have to be patient with this one and hope for the best.",
    "The uncertainty surrounding Roman Abramovich’s ownership of Chelsea deepened Thursday when the Premier League club abruptly halted plans to build a new stadium, citing an unfavorable climate to invest in the 500 million pound ($665 million) project.The announcement comes after it emerged the Russian billionaire was yet to have his British visa renewed amid a crackdown by authorities on associates of Russian President Vladimir Putin. Chelsea is yet to comment on the future at the Premier League club for Abramovich, who this week flew into Israel to receive Israeli citizenship.",
    "Saudi Arabia is backing a planned new 14-nation football federation in Asia aimed at boosting some of the region’s emerging nations in the game.Presidents of football federations in South and West Asia met in Jeddah on Thursday with a senior FIFA official attending as an observer to start the formal process of establishing a new organisation to be known as the South West Asian Football Federation (SWAFF).The nations represented at the meeting included Pakistan, Sri Lanka, India, Nepal, Bangladesh, Saudi Arabia, Bahrain, Maldives, Yemen, Oman, Kuwait, and United Arab Emirates.“The nations of South and West Asia want to work with each other to grow football in the region, and to compete on a more equal playing field at future World Cup competitions and international tournaments,” Dr Adel Ezzat, President and Chairman of the Saudi Arabian Football Federation said in a press release issued by organisers.The Asian Football Confederation (AFC) already includes five zonal federations for West, South, Central, East and Southeast Asia countries under its continental umbrella.AFC President Shaikh Salman bin Ebrahim Al Khalifa met with Ezzat to discuss the formation of SWAFF last week, the AFC said in statement.“We had an open and honest discussion on the formation of SWAFF and I made it clear to Mr Ezzat that the AFC had no objection … as long as it remains as a football body outside of the AFC’s zonal structure,” Shaikh Salman, a member of the Bahraini royal family, was quoted as saying.“SWAFF can come into existence on the lines of the Arab Gulf Cup Football Federation or the Union of Arab Football Associations, which are not part of the AFC but serve the greater purpose of bringing together many Gulf and Arab countries for the sole purpose of football development.“I am happy to note that Mr Ezzat agreed and confirmed that the establishment of SWAFF will not have any impact on the AFC’s five existing zones … and their current composition.”Just five of the 47 member nations of the AFC qualified for this year’s World Cup in Russia – Saudi Arabia, Iran, South Korea, Japan and Australia, which joined from Oceania in 2006.",
]
print("===================")
X_test = tfidf_vec.transform(test)
lda_test = lda.transform(X_test)
print(lda_test)

"""
機械学習_潜在意味解析_pythonで実装
https://dev.classmethod.jp/machine-learning/2017ad_20171221_lda_python/#sec%EF%BC%92
"""