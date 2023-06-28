#with open("test.txt","a") as myfile:
    #myfile.write("appended text")
    
import nltk
 
def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})
 
#print(format_sentence("The cat is very cute"))

pos = []
with open("/Users/RCM/anaconda3/lib/python3.6/site-packages/pos_tweets.txt", "r", encoding='iso-8859-1') as myfile:
    for e in myfile: 
        pos.append([format_sentence(e), 'pos'])
 
neg = []
with open("/Users/RCM/anaconda3/lib/python3.6/site-packages/neg_tweets.txt", "r", encoding='iso-8859-1') as myfile2:
    for i in myfile2: 
        neg.append([format_sentence(i), 'neg'])
 
# next, split labeled data into the training and test data
training = pos[:int((.8) * len(pos))] + neg[:int((.8) * len(neg))]
test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]

from nltk.classify import NaiveBayesClassifier
 
classifier = NaiveBayesClassifier.train(training)
myfile.close()
myfile2.close()

def diary():
	with open("diary.txt", "r")as myfiled:
		entry = myfiled.read().replace('\n','')
		myfiled.close()
	return entry

def analysis(entry):
	negval = 0.0
	posval = 0.0
	aentry = entry.split(".")
	for i in aentry:
		if classifier.classify(format_sentence(i)) == "neg":
			negval += 1.0
		else:
			posval += 1.0
	percentdif = abs((posval * 100.0 / float(len(aentry))) - (negval * 100.0 / float(len(aentry))))
	if percentdif <= 20.0:
		val = "neutral"
	elif percentdif > 20.0:
		if posval > negval:
			val = "positive"
		elif negval > posval:
			val = "negative"
	return val

def main():
	entry = diary()
	value = analysis(entry)
	print(value)
	return value
main()
# YOU NEED TO HARDCODE THE DIRECTORY FOR pos_tweets.txt AND neg_tweets.txt ON LINES 12 AND 17