# to write data in excel file :
import pandas as pd

Data = {'Name':['PPA','LB','python'],'Duration':[4,3,4]}
df= pd.DataFrame(Data,)
writer=pd.ExcelWriter('pandas0.xlsx',engine='xlsxwriter' )
df.to_excel(writer,sheet_name='Sheet1')
file= writer.save()