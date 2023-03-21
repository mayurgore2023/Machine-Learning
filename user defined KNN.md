# User defined KNN Algorithm:


# finding out euclidian distance:
def euc(a,b):
    return distance.euclidean(a,b)

# class include te testing/training and finding out nearest neighbour logic
class MarvellousKNN():

# Training logic:
    def fit(self,TrainingData,TrainingTarget):
        self.TrainingData = TrainingData
        self.TrainingTarget =TrainingTarget

#Testing logic:
    def predict(self,TestData):
        predictions = []
        for row in TestData:
            lebal = self.Closest(row)
            predictions.append(lebal)
        return predictions

# logic for finding nearest neighbour:
    def Closest(self,row):
        bestdistance = euc(row,self.TrainingData[0])
        bestindex = 0
        for i in range (1,len(self.TrainingData)):
            dist = euc(row,self.TrainingData[i])
            if dist < bestdistance:
                bestdistance = dist
                bestindexv = i
            return  self.TrainingTarget[bestindex]
