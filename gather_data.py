import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re

def df_from_fbref(year='current'):
    """
    This function reads in data about MLS matches given a year (season).
    
    Input: year for which we want data. The default is to get the current season's data.
    Note: This code was written in September 2023, so using the 'current' option will get the 2023 season.
    If 2023 is used as an input, then the function will fail.
    """
    if year == 'current':
        link = 'https://fbref.com/en/comps/22/schedule/Major-League-Soccer-Scores-and-Fixtures'
    else:
        link = f"https://fbref.com/en/comps/22/{year}/schedule/{year}-Major-League-Soccer-Scores-and-Fixtures"
        
    raw = requests.get(link)
    if raw.status_code != 200:
        print('Error fetching data from FBref.com. Status code:',raw.status_code)
        return
    else:
        bs = BeautifulSoup(raw.text, features='lxml')
        rows = bs.find('table').find('tbody').find_all('tr')
        
        # Put data into array
        data_array = []
        for row in rows:
            data_array.append([entry.text for entry in row.find_all(['th','td'])])
        
        # Put array into pandas DataFrame
        df = pd.DataFrame(data=data_array, columns=['round','day','date','local_time','home_team','home_xg','score',
                                                   'away_xg','away_team','attendance','stadium','referee','match_report',
                                                   'comment'])

        # Remove rows that are just repeat of the header
        # If 'score' column contains word 'Score' instead of a score, it should be removed
        df = df[~df.score.str.contains("Score")]
        
        # Drop any games without scores (these games have not happened yet)
        df.drop(df[df['score'] == ''].index, inplace=True)
        
        # Separate score into home and away goals
        hs_pattern = re.compile('\d*–')
        df['home_score'] = df['score'].apply(lambda x: hs_pattern.findall(x)[0][:-1])
        as_pattern = re.compile('–\d*')
        df['away_score'] = df['score'].apply(lambda x: as_pattern.findall(x)[0][1:])
        
        # Change 'local_time' to a decimal
        df['local_time'] = df['local_time'].apply(lambda x: int(x.strip().split(":")[0])+int(x.strip().split(":")[1])/60)
        
        # Add in a 'round' column if one does not exist
        if 'round' not in df.columns:
            df['round'] = len(df)*['Regular Season']
        
        # Remove columns that are not needed (like xg, referee, match_report)
        df = df[['round','day','date','local_time','home_team','home_score','away_score','away_team','attendance','stadium']]
        
        # Add in information about stadiums
        if year == 'current':
            sheet_name = '2023'
        else:
            sheet_name = str(year)       
        stad_df = pd.read_excel('mls_stadiums.xlsx', sheet_name=sheet_name)
        
        # Change teams to integer IDs
        df['home_team'] = df['home_team'].apply(lambda x: stad_df[stad_df['Team']==x]['Team_ID'].iloc[0])
        df['away_team'] = df['away_team'].apply(lambda x: stad_df[stad_df['Team']==x]['Team_ID'].iloc[0])
        
        # Latitude and longitude of stadium
        df['latitude'] = df['stadium'].apply(lambda x: stad_df[stad_df['Stadium']==x]['Latitude'].iloc[0])
        df['longitude'] = df['stadium'].apply(lambda x: stad_df[stad_df['Stadium']==x]['Longitude'].iloc[0])
        # Attendance divided by capacity
        df['attendance'] = [0 if x=='' else int(x.replace(",","")) for x in df['attendance']]
        
        # Create new column called 'playoff' which indicates whether a match was in the playoffs
        df['playoff'] = df['round'].apply(lambda x: 0 if x=='Regular Season' else 1)
        
        # Add missing attendance data from excel file
        missing_att = pd.read_excel('missing_attendance.xlsx', sheet_name='fill')
        
        # Get indices of matches with attendance = 0
        df_zero_ind = df[df['attendance']==0].index
        
        # Find matches with missing attendance in the missing attendance list
        for ind in df_zero_ind:
            match_ind = missing_att[(missing_att['date']==df.loc[ind,'date'])&\
                                    (missing_att['home_team']==df.loc[ind,'home_team'])&\
                                    (missing_att['away_team']==df.loc[ind,'away_team'])].index
            
            if len(match_ind)==1:
                df.loc[ind,'attendance'] = missing_att.loc[match_ind[0],'attendance']
                
                # Correct the stadium name if stadium name is wrong
                if pd.notna(missing_att.loc[match_ind[0],'stad_fix']):
                    df.loc[ind,'stadium'] = missing_att.loc[match_ind[0],'stad_fix']
        
        df['att_div_capacity'] = df.apply(lambda x: x.attendance / stad_df[stad_df['Stadium']==x.stadium]['Capacity'].iloc[0], axis=1)
        # Determine if either team in the match is the tenant of the stadium
        df['real_home_team'] = df.apply(lambda x: 1 if (stad_df[stad_df['Stadium']==x.stadium]['Team_ID'].iloc[0]==x.home_team)|\
                                        (stad_df[stad_df['Stadium']==x.stadium]['Team_ID'].iloc[0]==x.away_team) else 0, axis=1)
        
        # Read in information about conferences and rivalries
        conf_riv_df = pd.read_excel('mls_rivals.xlsx', sheet_name=sheet_name)
        
        # Get conference of each home team and away team
        home_conf = [conf_riv_df[conf_riv_df['Team_ID']==x]['Conference'].iloc[0] for x in df['home_team']]
        away_conf = [conf_riv_df[conf_riv_df['Team_ID']==x]['Conference'].iloc[0] for x in df['away_team']]
        
        # Create new column that indicates if both teams are from the same conference
        df['same_conf'] = [1 if x==y else 0 for x,y in zip(home_conf,away_conf)]
        
        # Change the 'Rivals' column so that each entry is a list
        conf_riv_df['Rivals'] = conf_riv_df.Rivals.apply(eval)
        
        # Make new column that indicates whether the teams are rivals
        df['rivals'] = df.apply(lambda x: 1 if x.away_team in conf_riv_df[conf_riv_df['Team_ID']==x.home_team]['Rivals'].iloc[0] else 0,\
                                axis=1)
        
        return df
        
def add_weather_info(data):
    """
    This function takes in a Pandas DataFrame of matches and adds in weather information from open-meteo.com.
    The input DataFrame must have the following columns:
    date: Date of the match formatted as yyyy-mm-dd
    local_time: Local time as a decimal (24 hour format so that 1:00 pm is 13)
    latitude: latitude of stadium
    longitude: longitude of stadium
    """
    # Create lists to hold weather data
    temperature = []
    rain = []
    snow = []
    cloud = []
    windspeed = []
    windgust = []
    rain_sum = []
    snow_sum = []
    
    url1 = 'https://archive-api.open-meteo.com/v1/archive?latitude='
    url2 = '&longitude='
    url3 = '&start_date='
    url4 = '&end_date='
    url5 = '&hourly=temperature_2m,rain,snowfall,cloudcover,windspeed_10m,windgusts_10m&temperature_unit=fahrenheit'
    
    # Loop over each match
    for ind in data.index:
        date = data['date'][ind]
        loc_time = int(np.floor(data['local_time'][ind]))
        lat = data['latitude'][ind]
        long = data['longitude'][ind]
        full_url = f"{url1}{lat}{url2}{long}{url3}{date}{url4}{date}{url5}"
        
        weather_req = requests.get(full_url)
        
        # If request fails, append None for weather data
        if weather_req.status_code != 200:
            print(f"Request failed for index {ind}. Status code = {weather_req.status_code}")
            temperature.append(None)
            rain.append(None)
            snow.append(None)
            cloud.append(None)
            windspeed.append(None)
            windgust.append(None)
            rain_sum.append(None)
            snow_sum.append(None)
        else:
            # Using nearest hour to kick-off time, add weather values to lists
            temperature.append(weather_req.json()['hourly']['temperature_2m'][loc_time])
            rain.append(weather_req.json()['hourly']['rain'][loc_time])
            snow.append(weather_req.json()['hourly']['snowfall'][loc_time])
            cloud.append(weather_req.json()['hourly']['cloudcover'][loc_time])
            windspeed.append(weather_req.json()['hourly']['windspeed_10m'][loc_time])
            windgust.append(weather_req.json()['hourly']['windgusts_10m'][loc_time])
            # For rain and snow, sum up the rain and snow since midnight before kick-off
            try:
                rain_sum.append(np.sum(weather_req.json()['hourly']['rain'][:loc_time]))
            except:
                rain_sum.append(None)
            try:
                snow_sum.append(np.sum(weather_req.json()['hourly']['snowfall'][:loc_time]))
            except:
                snow_sum.append(None)

    # Add data to the DataFrame
    data_new = data.copy()
    data_new['temperature'] = temperature
    data_new['rain'] = rain
    data_new['snow'] = snow
    data_new['cloudcover'] = cloud
    data_new['windspeed'] = windspeed
    data_new['windgust'] = windgust
    data_new['rain_sum'] = rain_sum
    data_new['snow_sum'] = snow_sum
    
    return data_new