# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
from sklearn import preprocessing
from flask_basicauth import BasicAuth
import csv


import traceback
import pandas as pd
import numpy as np
import json

# Your API definition
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'admin'
app.config['BASIC_AUTH_PASSWORD'] = 'password'
basic_auth = BasicAuth(app)

def csvtodict(filename):
    data={}
    with open(filename) as fin:
        reader=csv.reader(fin, skipinitialspace=True)
        for row in reader:
            if not (row):
                continue
            data[row[0]] = int(row[1])
    return data
#input_file = csv.DictReader(open(file))
#print(input_file)
#print(csvtodict(file))

file='Rail/Dictionary/RandomForest/Version1/TrackID_dictionary.csv'
#{'S70P0R01': 0, 'S70P0R05': 1, 'W40P0G01': 2, 'W40P0R01': 3, 'W40P0R02': 4, 'W40P0R03': 5, 'W40P0W09': 6, 'W50P0R02': 7, 'W50P0R03': 8, 'W50P0R05': 9, 'W50P0R06': 10, 'W50P0W15': 11, 'RS81-S80P0R01': 12, 'RS92-S90P0R02': 13, 'RS95-S90P0R05': 14, 'S70P0T01': 15, 'S80P0T03': 16, 'TS71-S70P0T01': 17, 'TS72-S70P0T02': 18, 'TS73-S70P0T03': 19, 'TS74-S70P0T04': 20, 'TS75-S70P0T05': 21, 'TW41-W40P0T01': 22, 'TW42-W40P0T02': 23, 'TW43-W40P0T03': 24, 'TW51-W50P0T01': 25, 'W51T01TA': 26, 'W51T04TA': 27, 'undefined': 28}

TrackID_dictionary_rail = csvtodict(file)
#{'1': 0, '2': 1, '34': 2, 'AO 11': 3, 'AO 13': 4, 'AO 15': 5, 'BG01': 6, 'Y022': 7, 'Y023': 8, 'undefined': 9}

file='Rail/Dictionary/RandomForest/Version1/AnchorPattern_dictionary.csv'
AnchorPattern_dictionary_rail = csvtodict(file)

file = 'Rail/Dictionary/RandomForest/Version1/MegaWorkBlock_dictionary.csv'
MegaWorkBlock_dictionary_rail = csvtodict(file)
#{'No': 0, 'Yes': 1}
file ='Rail/Dictionary/RandomForest/Version1/ShadowWorkBlock_dictionary.csv'
ShadowWorkBlock_dictionary_rail = csvtodict(file)
#{'No': 0, 'Yes': 1}
file ='Rail/Dictionary/RandomForest/Version1/SplitWorkBlock_dictionary.csv'
SplitWorkBlock_dictionary_rail = csvtodict(file)
#{'No': 0, 'Yes': 1}
#file ='Rail/Dictionary/RandomForest/Version1/TieType_dictionary.csv'
#TieType_dictionary_rail = csvtodict(file)
#{'CONCRETE': 0, 'WOOD': 1}
#{'Every 2 nd': 0, 'Every 3 rd': 1, 'Every 4 th': 2, 'Every Tie': 3, 'Pattern 10': 4, 'Pattern 6': 5, 'Pattern 8': 6}
file ='Rail/Dictionary/RandomForest/Version1/SpikePattern_dictionary.csv'
SpikePattern_dictionary_rail = csvtodict(file)
#{'3 Spiked': 0, '4 Spiked': 1, '5 Spiked': 2, '6 Spiked': 3, 'B': 4, 'C': 5, 'D': 6, 'NONE': 7}
#TieNumberType_dictionary_rail = {'None': 0, 'Tie Type 1': 1, 'Tie Type 2': 2}
#PadInsulatorType_dictionary_rail  = {'None': 0, 'Pads': 1}
file ='Rail/Dictionary/RandomForest/Version1/FastenerTypeTie_dictionary.csv'
FastenerTypeTie_dictionary_rail = csvtodict(file)
#{'CLIP_RAIL': 0, 'None': 1}
file ='Rail/Dictionary/RandomForest/Version1/PlateChangeOutRequired_dictionary.csv'
PlateChangeOutRequired_dictionary_rail = csvtodict(file)
#{'No': 0, 'None': 1}
file = 'Rail/Dictionary/RandomForest/Version1/SubDivision_dictionary.csv'
SubDivision_dictionary_rail = csvtodict(file)
#{'ALBREDA': 0, 'ASHCROFT': 1, 'BRAZEAU': 2, 'CAMROSE': 3, 'CLEARWATER': 4, 'FORT FRANCES': 5, 'NEENAH': 6, 'RAINY': 7, 'RIVERS': 8, 'SPRAGUE': 9, 'SUPERIOR': 10, 'THREE HILLS': 11, 'VEGREVILLE': 12, 'YALE': 13}

file='Rail/Dictionary/RandomForest/Version1/RailType_dictionary.csv'
RailType_dictionary_rail= csvtodict(file)
#{'New': 0, 'Old': 1}
file ='Rail/Dictionary/RandomForest/Version1/CwrTerritory_dictionary.csv'
CwrTerritory_dictionary_rail = csvtodict(file)
#{'No': 0, 'Yes': 1}
file ='Rail/Dictionary/RandomForest/Version1/DestressingMethod_dictionary.csv'
DestressingMethod_dictionary_rail = csvtodict(file)
#{'HEATERS': 0, 'PULLERS': 1}
#file ='Rail/Dictionary/RandomForest/Version1/workBlockPlannedMinutes_dictionary.csv'
#workBlockPlannedMinutes_dictionary= csvtodict(file)
#{'60': 0, '180': 1, '200': 2, '240': 3, '300': 4, '360': 5, '420': 6, '440': 7, '480': 8, '600': 9}
file = 'Rail/Dictionary/RandomForest/Version1/ClosureType_dictionary.csv'
ClosureType_dictionary_rail= csvtodict(file)
#{'No': 0, 'Yes': 1}
file = 'Rail/Dictionary/RandomForest/Version1/GangId_dictionary.csv'
GangId_dictionary_rail = csvtodict(file)

#GangId_dictionary_Tie = {'S70P0S03': 0, 'S70P0T01': 1, 'S72T03WA': 2, 'W42T04TA': 3, 'W43T03TC': 4, 'W50P0S05': 5, 'W50P0S06': 6, 'W50P0T02': 7, 'W50P0T03': 8, 'W52T07TC': 9, 'W53T0100': 10, 'W53T01TA': 11, 'W53T01TE': 12, 'W53T02TC': 13, 'W53T02TE': 14, 'W53T02TJ': 15, 'W53T03TA': 16, 'W53T03TC': 17, 'W53T03TD': 18, 'W53T04TA': 19, 'W53T05TA': 20, 'W53T05WA': 21, 'W53T06TB': 22, 'W53T06TD': 23, 'W53T06TE': 24, 'W53T06TK': 25, 'W53T06WA': 26, 'W54T02TB': 27, 'W54T02TC': 28, 'W54T02TD': 29, 'W54T02TI': 30, 'W54T04TD': 31, 'W54T05TB': 32, 'W54T0600': 33, 'W54T09TC': 34, 'W55T01TD': 35}
#TrackID_dictionary_Tie = {'01': 0, '02': 1, '04S': 2, 'CG12': 3, 'DF19': 4, 'DF21': 5, 'KC14': 6, 'KC16': 7, 'LD00': 8, 'N822': 9, 'NN14': 10, 'PA03': 11, 'PA04': 12, 'PA05': 13, 'PA07': 14, 'PA10': 15, 'PA12': 16, 'PA13': 17, 'PF00': 18, 'PF11': 19, 'PF12': 20, 'PF42': 21, 'PF50': 22, 'PS13': 23, 'PS15': 24, 'PS19': 25, 'PS24': 26, 'PV04': 27, 'RF53': 28, 'T014': 29, 'T020': 30, 'T021': 31, 'XO11': 32, 'YG99': 33}
#MegaWorkBlock_dictionary_Tie = {'No': 0, 'Yes': 1}
#ShadowWorkBlock_dictionary_Tie = {'No': 0, 'Yes': 1}
#SplitWorkBlock_dictionary_Tie = {'No': 0, 'Yes': 1}
#TieType_dictionary_Tie = {'CONCRETE': 0, 'WOOD': 1}
#AnchorPattern_dictionary_Tie = {'Pattern 10': 0, 'Pattern 6': 1, 'Pattern 8': 2}
#SpikePattern_dictionary_Tie = {'A': 0, 'B': 1, 'D': 2}
#TieNumberType_dictionary_Tie = {'Tie Type 1': 0, 'none': 1, 'Tie Type 2': 2}
#PadInsulatorType_dictionary  = {'None': 0, 'Pads': 1}
#FastenerTypeTie_dictionary_Tie = {'CLIP_RAIL': 0, 'none': 1, 'None': 1}
#PlateChangeOutRequired_dictionary_Tie = {'No': 0, 'None': 1} Note : Feature not there
#SubDivision_dictionary_Tie = {'ASHCROFT': 0, 'BULKLEY': 1, 'CHETWYND': 2, 'CLEARWATER': 3, 'KASHABOWIE': 4, 'LETELLIER': 5, 'LILLOOET': 6, 'LUMBY': 7, 'NEENAH': 8, 'OKANAGAN': 9, 'PRINCE GEORGE': 10, 'SKEENA': 11, 'SLAVE LAKE': 12, 'SQUAMISH': 13, 'THREE HILLS': 14, 'VEGREVILLE': 15, 'WAUKESHA': 16, 'YALE': 17}
#OperationNumber_dictionary_Tie = {'410': 0, '510': 1, '710': 2}
file='Tie/Dictionary/RandomForest/Version1/OperationNumber_dictionary.csv'
OperationNumber_dictionary_Tie=csvtodict(file)
file='Tie/Dictionary/RandomForest/Version1/TrackID_dictionary.csv'
#{'S70P0R01': 0, 'S70P0R05': 1, 'W40P0G01': 2, 'W40P0R01': 3, 'W40P0R02': 4, 'W40P0R03': 5, 'W40P0W09': 6, 'W50P0R02': 7, 'W50P0R03': 8, 'W50P0R05': 9, 'W50P0R06': 10, 'W50P0W15': 11, 'RS81-S80P0R01': 12, 'RS92-S90P0R02': 13, 'RS95-S90P0R05': 14, 'S70P0T01': 15, 'S80P0T03': 16, 'TS71-S70P0T01': 17, 'TS72-S70P0T02': 18, 'TS73-S70P0T03': 19, 'TS74-S70P0T04': 20, 'TS75-S70P0T05': 21, 'TW41-W40P0T01': 22, 'TW42-W40P0T02': 23, 'TW43-W40P0T03': 24, 'TW51-W50P0T01': 25, 'W51T01TA': 26, 'W51T04TA': 27, 'undefined': 28}

TrackID_dictionary_Tie = csvtodict(file)
#{'1': 0, '2': 1, '34': 2, 'AO 11': 3, 'AO 13': 4, 'AO 15': 5, 'BG01': 6, 'Y022': 7, 'Y023': 8, 'undefined': 9}

file='Tie/Dictionary/RandomForest/Version1/AnchorPattern_dictionary.csv'
AnchorPattern_dictionary_Tie = csvtodict(file)

#file = 'Tie/Dictionary/RandomForest/Version1/MegaWorkBlock_dictionary.csv'
#MegaWorkBlock_dictionary_Tie = csvtodict(file)
#{'No': 0, 'Yes': 1}
#file ='Tie/Dictionary/RandomForest/Version1/ShadowWorkBlock_dictionary.csv'
#ShadowWorkBlock_dictionary_Tie = csvtodict(file)
#{'No': 0, 'Yes': 1}
#file ='Tie/Dictionary/RandomForest/Version1/SplitWorkBlock_dictionary.csv'
#SplitWorkBlock_dictionary_Tie = csvtodict(file)
#{'No': 0, 'Yes': 1}
file ='Tie/Dictionary/RandomForest/Version1/TieType_dictionary.csv'
print("tie type dict")
TieType_dictionary_Tie = csvtodict(file)
#{'CONCRETE': 0, 'WOOD': 1}
#{'Every 2 nd': 0, 'Every 3 rd': 1, 'Every 4 th': 2, 'Every Tie': 3, 'Pattern 10': 4, 'Pattern 6': 5, 'Pattern 8': 6}
file ='Tie/Dictionary/RandomForest/Version1/SpikePattern_dictionary.csv'
SpikePattern_dictionary_Tie = csvtodict(file)
#{'3 Spiked': 0, '4 Spiked': 1, '5 Spiked': 2, '6 Spiked': 3, 'B': 4, 'C': 5, 'D': 6, 'NONE': 7}
#TieNumberType_dictionary_Tie = {'None': 0, 'Tie Type 1': 1, 'Tie Type 2': 2}
file ='Tie/Dictionary/RandomForest/Version1/TieNumberType_dictionary.csv'
TieNumberType_dictionary_Tie = csvtodict(file)
#PadInsulatorType_dictionary  = {'None': 0, 'Pads': 1}
file ='Tie/Dictionary/RandomForest/Version1/FastenerTypeTie_dictionary.csv'
FastenerTypeTie_dictionary_Tie = csvtodict(file)
#{'CLIP_RAIL': 0, 'None': 1}
#file ='Tie/Dictionary/RandomForest/Version1/PlateChangeOutRequired_dictionary.csv'
#PlateChangeOutRequired_dictionary_Tie = csvtodict(file)
#{'No': 0, 'None': 1}
file = 'Tie/Dictionary/RandomForest/Version1/SubDivision_dictionary.csv'
SubDivision_dictionary_Tie = csvtodict(file)
#{'ALBREDA': 0, 'ASHCROFT': 1, 'BRAZEAU': 2, 'CAMROSE': 3, 'CLEARWATER': 4, 'FORT FRANCES': 5, 'NEENAH': 6, 'RAINY': 7, 'RIVERS': 8, 'SPRAGUE': 9, 'SUPERIOR': 10, 'THREE HILLS': 11, 'VEGREVILLE': 12, 'YALE': 13}

#file='Tie/Dictionary/RandomForest/Version1/RailType_dictionary.csv'
#RailType_dictionary= csvtodict(file)
#{'New': 0, 'Old': 1}
#file ='Tie/Dictionary/RandomForest/Version1/CwrTerritory_dictionary.csv'
#CwrTerritory_dictionary_Tie = csvtodict(file)
#{'No': 0, 'Yes': 1}
#file ='Tie/Dictionary/RandomForest/Version1/DestressingMethod_dictionary.csv'
#DestressingMethod_dictionary_Tie = csvtodict(file)
#{'HEATERS': 0, 'PULLERS': 1}
#file ='Tie/Dictionary/RandomForest/Version1/workBlockPlannedMinutes_dictionary.csv'
#workBlockPlannedMinutes_dictionary= csvtodict(file)
#{'60': 0, '180': 1, '200': 2, '240': 3, '300': 4, '360': 5, '420': 6, '440': 7, '480': 8, '600': 9}
#file = 'Tie/Dictionary/RandomForest/Version1/ClosureType_dictionary.csv'
#ClosureType_dictionary= csvtodict(file)
#{'No': 0, 'Yes': 1}
file = 'Tie/Dictionary/RandomForest/Version1/GangId_dictionary.csv'
GangId_dictionary_Tie = csvtodict(file)


def data_preprocessing_rail(Rail_df):
     Rail_df[Rail_df.select_dtypes(['object']).columns] = Rail_df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
     #col_list = ['ActivityParentId', 'ActivityId' , 'PlanId', 'Date', 'ConfidenceLevel', 'ProductionType', 'Version','CwrTerritory','RailType','ClosureType','DestressingMethod','OrderNumber']
     #col_list = ['ProductionType', 'Version','CwrTerritory','RailType','ClosureType','DestressingMethod','OrderNumber']
     #tier_df=tier_df.drop(columns=col_list)
     Rail_df['Miles'] = abs(Rail_df['WBMileFrom'] - Rail_df['WBMileTo'])

     encode = preprocessing.LabelEncoder()
     Rail_df['GangId'] = GangId_dictionary_rail[Rail_df['GangId'].values[0]]
     Rail_df['TrackId'] = TrackID_dictionary_rail[Rail_df['TrackId'].values[0]]
     try:
        Rail_df['MegaWorkBlock']=Rail_df['MegaWorkBlock'].cat.add_categories('No')
     except:
        print("")
     Rail_df['MegaWorkBlock'].fillna('No', inplace=True)
     Rail_df['MegaWorkBlock']=MegaWorkBlock_dictionary_rail[Rail_df['MegaWorkBlock'].values[0]]
     try:
        Rail_df['ShadowWorkBlock']=Rail_df['ShadowWorkBlock'].cat.add_categories('No')
     except:
        print("")
     Rail_df['ShadowWorkBlock'].fillna('No', inplace=True)
     Rail_df['ShadowWorkBlock']=ShadowWorkBlock_dictionary_rail[Rail_df['ShadowWorkBlock'].values[0]]
     #Rail_df['SplitWorkBlock'].fillna(0, inplace=True)
     Rail_df['SplitWorkBlock']=SplitWorkBlock_dictionary_rail[Rail_df['SplitWorkBlock'].values[0]]
     #Rail_df['TieType']=TieType_dictionary_rail[Rail_df['TieType'].values[0]]
     Rail_df['RailType']=RailType_dictionary_rail[Rail_df['RailType'].values[0]]
     Rail_df['AnchorPattern']=AnchorPattern_dictionary_rail[Rail_df['AnchorPattern'].values[0]]
     Rail_df['SpikePattern']=SpikePattern_dictionary_rail[Rail_df['SpikePattern'].values[0]]
     Rail_df['CwrTerritory']=CwrTerritory_dictionary_rail[Rail_df['CwrTerritory'].values[0]]
     Rail_df['DestressingMethod']=DestressingMethod_dictionary_rail[Rail_df['DestressingMethod'].values[0]]
     Rail_df['workBlockPlannedMinutes']=Rail_df['workBlockPlannedMinutes']
     Rail_df['ClosureType']=ClosureType_dictionary_rail[Rail_df['ClosureType'].values[0]]
     #tier_df['NumberofInsulatedJoints'].fillna(0, inplace=True)
     #tier_df['NumberofCompromiseRails'].fillna(0, inplace=True)
     #tier_df['TravelTimeDuringBlocks'].fillna(0, inplace=True)
     #tier_df['TieNumberType'].fillna('None', inplace=True)
     #Rail_df['TieNumberType']=TieNumberType_dictionary_rail[Rail_df['TieNumberType'].values[0]]
     #tier_df['NumberofTampers'].fillna(0, inplace=True)
     #tier_df['PadInsulatorType']=tier_df['PadInsulatorType'].cat.add_categories('None')
     #tier_df['PadInsulatorType'].fillna('None', inplace=True)
     #Rail_df['PadInsulatorType']=PadInsulatorType_dictionary_rail[Rail_df['PadInsulatorType'].values[0]]
     #tier_df['FastenerTypeTie']=tier_df['FastenerTypeTie'].cat.add_categories('None')
     #tier_df['FastenerTypeTie'].fillna('None', inplace=True)
     Rail_df['FastenerTypeTie']=FastenerTypeTie_dictionary_rail[Rail_df['FastenerTypeTie'].values[0]]
     #tier_df['PlateChangeOutRequired']=tier_df['PlateChangeOutRequired'].cat.add_categories('None')
     #tier_df['PlateChangeOutRequired'].fillna('None', inplace=True)
     Rail_df['PlateChangeOutRequired']=PlateChangeOutRequired_dictionary_rail[Rail_df['PlateChangeOutRequired'].values[0]]
     Rail_df['SubDivision']=SubDivision_dictionary_rail[Rail_df['SubDivision'].values[0]]
     #tier_df['NumberofTransitionRails'].fillna(0, inplace=True)
     Rail_df['OperationNumber']=Rail_df.OperationNumber.str.extract('(^\d*)')
     Rail_df = Rail_df[Rail_df['OperationNumber'] != ""]
     Rail_df['OperationNumber'] = Rail_df['OperationNumber'].astype('int32')
     return Rail_df

def data_preprocessing_Tie(tier_df):
     tier_df["WBMileFrom"] = tier_df["WBMileFrom"].astype("float")
     tier_df["WBMileTo"] = tier_df["WBMileTo"].astype("float")
     tier_df['Miles'] = abs(tier_df['WBMileFrom'] - tier_df['WBMileTo'])
     tier_df.drop(['ShadowWorkBlock', 'SplitWorkBlock', 'TravelTimeDuringBlocks', 'NumberofInsulatedJoints', 'MegaWorkBlock', 'PlateChangeOutRequired', 'NumberofTampers',
                   'NumberofTransitionRails', 'NumberofCompromiseRails', 'PadInsulatorType', 'WBMileFrom', 'WBMileTo'],axis=1,inplace=True)
     tier_df[tier_df.select_dtypes(['object']).columns] = tier_df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
     encode = preprocessing.LabelEncoder()
     tier_df['GangId'] = GangId_dictionary_Tie[tier_df['GangId'].values[0]]
     tier_df['TrackId'] = TrackID_dictionary_Tie[tier_df['TrackId'].values[0]]
     #tier_df['MegaWorkBlock'].fillna('No', inplace=True)
     #tier_df['MegaWorkBlock']=MegaWorkBlock_dictionary_Tie[tier_df['MegaWorkBlock'].values[0]]
     #tier_df['ShadowWorkBlock'].fillna('No', inplace=True)
     #tier_df['ShadowWorkBlock']=ShadowWorkBlock_dictionary_Tie[tier_df['ShadowWorkBlock'].values[0]]
     #tier_df['SplitWorkBlock'].fillna('No', inplace=True)
     #tier_df['SplitWorkBlock']=SplitWorkBlock_dictionary_Tie[tier_df['SplitWorkBlock'].values[0]]
     tier_df['TieType']=TieType_dictionary_Tie[tier_df['TieType'].values[0]]
     tier_df['AnchorPattern']=AnchorPattern_dictionary_Tie[tier_df['AnchorPattern'].values[0]]
     tier_df['SpikePattern']=SpikePattern_dictionary_Tie[tier_df['SpikePattern'].values[0]]
     #tier_df['NumberofInsulatedJoints'].fillna(0, inplace=True)
     #tier_df['NumberofCompromiseRails'].fillna(0, inplace=True)
     #tier_df['TravelTimeDuringBlocks'].fillna(0, inplace=True)
     #tier_df['TieNumberType'].fillna('None', inplace=True)
     tier_df['TieNumberType']=TieNumberType_dictionary_Tie[tier_df['TieNumberType'].values[0]]
     #tier_df['NumberofTampers'].fillna(0, inplace=True)
     #tier_df['PadInsulatorType']=tier_df['PadInsulatorType'].cat.add_categories('None')
     #tier_df['PadInsulatorType'].fillna('None', inplace=True)
     #tier_df['PadInsulatorType']=PadInsulatorType_dictionary_Tie[tier_df['PadInsulatorType'].values[0]]
     #tier_df['FastenerTypeTie']=tier_df['FastenerTypeTie'].cat.add_categories('None')
     #tier_df['FastenerTypeTie'].fillna('None', inplace=True)
     tier_df['FastenerTypeTie']=FastenerTypeTie_dictionary_Tie[tier_df['FastenerTypeTie'].values[0]]
     #tier_df['PlateChangeOutRequired']=tier_df['PlateChangeOutRequired'].cat.add_categories('None')
     #tier_df['PlateChangeOutRequired'].fillna('None', inplace=True)
     #tier_df['PlateChangeOutRequired']=PlateChangeOutRequired_dictionary_Tie[tier_df['PlateChangeOutRequired'].values[0]]
     tier_df['SubDivision']=SubDivision_dictionary_Tie[tier_df['SubDivision'].values[0]]
     #tier_df['NumberofTransitionRails'].fillna(0, inplace=True)
     #tier_df['OperationNumber']=tier_df.OperationNumber.str.extract('(^\d*)')
     #tier_df = tier_df[tier_df['OperationNumber'] != ""]
     tier_df['OperationNumber'] = tier_df['OperationNumber'].astype('category')
     tier_df['OperationNumber']=OperationNumber_dictionary_Tie[tier_df['OperationNumber'].values[0]]
     return tier_df


@app.route('/tie/predict', methods=['POST'])
@basic_auth.required
def predictTie():
    if tiemodel:
        try:
            json_ = request.json
            print(json_)
            query = pd.read_json(json.dumps(json_),orient='index')
            query.reset_index(level=0,inplace=True)
            columns = query['index']
            values = query.iloc[:,1:2]
            values = values.transpose()
            values.columns = columns
            tie_df_in = values
            columns=["SubDivision", "GangId", "OperationNumber", "TrackId", "TieType", "AnchorPattern", "SpikePattern","FastenerTypeTie", "MegaWorkBlock", "ShadowWorkBlock",  "SplitWorkBlock",  "TravelTimeDuringBlocks",  "TieNumberType", "workBlockPlannedMinutes", "TieDensity", "NumberofTampers", "NumberofInsulatedJoints","NumberofTransitionRails", "NumberofCompromiseRails",  "PlateChangeOutRequired",  "PadInsulatorType","WBMileFrom","WBMileTo"]
            tie_df=pd.DataFrame(columns = columns)
            for i in range(len(tie_df.columns)):
                tie_df[columns[i]] = tie_df_in[columns[i]]

            tie_df["WBMileFrom"] = tie_df["WBMileFrom"].astype("float")
            tie_df["WBMileTo"] = tie_df["WBMileTo"].astype("float")
            tie_df_X = data_preprocessing_Tie(tie_df)
            print(tie_df_X.head(12))
            prediction = tiemodel.predict(tie_df_X)
            return jsonify({'prediction': str(prediction[0])})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


@app.route('/rail/predict', methods=['POST'])
@basic_auth.required
def predictRail():
    if railmodel:
        try:
            json_ = request.json
            print(json_)
            #print(df = df.reindex(columns=list(json_[0].keys())))
            query = pd.read_json(json.dumps(json_),orient='index')
            query.reset_index(level=0,inplace=True)
            columns = query['index']
            values = query.iloc[:,1:2]
            values = values.transpose()
            values.columns = columns
            rail_df_in = values
            columns = ["SubDivision","GangId","OperationNumber","TrackId","MegaWorkBlock","ShadowWorkBlock","SplitWorkBlock","WBMileFrom","WBMileTo","AnchorPattern","SpikePattern","CwrTerritory","DestressingMethod","ClosureType","NumberofInsulatedJoints","NumberofTransitionRails","NumberofCompromiseRails","FastenerTypeTie","PlateChangeOutRequired","RailType","TravelTimeDuringBlocks","workBlockPlannedMinutes"]
            rail_df=pd.DataFrame(columns = columns)
            for i in range(len(rail_df.columns)):
                rail_df[columns[i]] = rail_df_in[columns[i]]
            rail_df["WBMileFrom"] = rail_df["WBMileFrom"].astype("float")
            rail_df["WBMileTo"] = rail_df["WBMileTo"].astype("float")
            rail_df_X = data_preprocessing_rail(rail_df)
            prediction = railmodel.predict(rail_df_X)
            return jsonify({'prediction': str(prediction[0])})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12346 # If you don't provide any port the port will be set to 12345

    railmodel = joblib.load("Rail/Model/RandomForest/Version1/finalized_model.sav") # Load "model.pkl"
    tiemodel = joblib.load("Tie/Model/RandomForest/Version1/finalized_model_tier_p3.pkl") # Load "model.pkl"

    print ('Model loaded')

    app.run(port=port, debug=True)