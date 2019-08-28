from sklearn import datasets
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts

wine = datasets.load_wine()

features = wine.data
labels = wine.target

# print ( "Number of entries: {} ".format(len(features)) )
# print ("\t".join( [x[:10] for x in wine.feature_names ] ) )
# for feature, label in zip(features, labels):
#     print ("{} {}".format("\t\t".join([str(x) for x in feature]), label))
# print ("\n")

tr_f, te_f, tr_l, te_l = tts(features, labels, test_size = 0.2)

print ( "Number of entries: {} ".format(len(tr_f)) )
print ("\t".join( [x[:10] for x in wine.feature_names ] ) )
for feature, label in zip(tr_f, tr_l):
    print ("{} {}".format("\t\t".join([str(x) for x in feature]), label))
print ("\n")


#clf = svm.SVC()
clf = svm.SVC(kernel = 'linear')
#clf = tree.DecisionTreeClassifier()
#clf = RandomForestClassifier()


# Train the features
clf.fit(tr_f, tr_l)

# Prediction
predict = clf.predict(te_f)
print (predict)

score=0
for i in range(len(predict)):
    if predict[i] == te_l[i]:
        score += 1
print (score/len(predict))




