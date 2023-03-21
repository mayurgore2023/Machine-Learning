#Different Way  write data in Dataframe:
import pandas as pd

#Dataframe using list:
Data = [1,2,3,4,5]
df = pd.DataFrame(Data)
print(df)

#Datafeame by creating colums:
Data = [['PPA',4],['LB',3],['Python',4]]
df = pd.DataFrame(Data,columns=['Name','Duration'])
print(df)
print()


#by using list in Dictonary:
Data ={'Name':['PPA','LB','Python'],'Duration':[4,3,4]}
df = pd.DataFrame(Data)
print(df)
print()

# usind dictionary in list:
Data = [{'Name':'PPA','Duration':4,'Fees':10500},{'Name':'Lb','Duration':3,'Fees':1500}]
df =pd.DataFrame(Data)
print(df)