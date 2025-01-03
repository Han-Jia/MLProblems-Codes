from sklearn import svm

clf = svm.SVC(kernel='linear')
clf.fit(X, Y)
