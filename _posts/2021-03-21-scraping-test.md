---
layout: post
title: "Web Scraping and Task Scheduling with BeautifulSoup and Airflow"
date: 2021-03-21
---

_[both the scraping and the DAG script are available on [github](https://github.com/bmill42/Tennis-Analytics)]_

Unlike the last post, which involved doing some exploration and prediction with a statis dataset, this post is more focused on the data engineering end of the data science stack. The goal here is to set up a basic web-scraping script to get tennis scores from ESPN and then schedule it to run daily using Airflow, all in Python. It's relatively brief, establishing a very simple pipeline that I can build on in the future.

First, we set up the scraper using the BeautifulSoup library, which provides a wonderfully user-friendly means of navigating HTML or XML structures and extracting the relevant information.


```python
from bs4 import BeautifulSoup
import re
import pandas as pd
from pathlib import Path
import urllib.request
import lxml
import datetime as dt
import sys
from time import sleep

data_file = Path("/Users/brian_miller/Data_Science/tennis_pipeline/match_data.xlsx")

if data_file.is_file():
    df = pd.read_excel(data_file).drop(columns=["Unnamed: 0"])
else:
    df = pd.DataFrame(columns=['date','winner','w_seed','loser','l_seed','score'])
```

For now, I'm just storing data in a spreadsheet, so to get started all that's needed is to check whether the spreadsheet already exists and either import it or set up an empty dataframe.

I'm scraping from ESPN's daily scoreboard, which can be accessed by passing the relevant date to the appropriate URL. Making a single daily scrape of tennis scores can be tricky since tournaments take place all around the world, in very different timezones. To deal with this, one could either do some more sophisticated scheduling based on specific tournmanent locations, daily match schedules, and so on, or one could simply wait to scrape until the following day (or even later) to make sure all matches are completed. For this simple script, I'm taking the latter approach and using yesterday's date for each scraping attempt. There's a bit of extra code to make sure the date that comes from `datetime` is zero-padded:


```python
yest = dt.date.today() - dt.timedelta(days=1)

yest_year = str(yest.year)
yest_month = str(yest.month)
if len(yest_month) == 1:
    yest_month = '0' + yest_month
yest_day = str(yest.day)
if len(yest_day) == 1:
    yest_day = '0' + yest_day

yest_string = yest_year + yest_month + yest_day
```

For testing purposes, the script can optionally take an argument on the command line that will be used in place of the actual date.

Then, it's straightforward to grab the HTML from the URL and parse it with BeautifulSoup (though a number of issues can come up here, discussed below).


```python
if len(sys.argv) > 1:
    yest_string = sys.argv[1]

link = "http://m.espn.com/general/tennis/dailyresults?date=" + yest_string
f = urllib.request.urlopen(link)

soup = BeautifulSoup(f, 'lxml')
```

Web scraping can be finnicky; I've chosen ESPN's scoreboard because it provides fairly clean HTML, but occasionally the page loads without displaying any matches. For now, when that happens the script just reloads the page after five seconds and tries again, making up to five attempts before giving up. While I'm using the mobile link, the URL request protocol I'm using currently gets the desktop version. That's an artifact of some earlier testing, but for now I'm not going to break what's working. There are other ways to go about this to try to ensure a more consistent response (like adding headers to the URL request to imitate a particular web browser, or using other scraping tools like Selenium), but since this is just a demonstration I've kept it simple. Eventually, the goal will be to scrape more detailed stats from each match, but that will involve using separate URLs to get the full box score for each match, or possibly even finding a different site to scrape from altogether.


```python
tries = 1
while len(soup.find_all(string=re.compile("No Matches Scheduled", re.I))) > 0:
    if tries < 6:

        print('no matches, trying again')

        sleep(5)

        link = "http://m.espn.com/general/tennis/dailyresults?date=" + yest_string
        f = urllib.request.urlopen(link)

        soup = BeautifulSoup(f, 'lxml')

        tries += 1
    else:
        print('could\'t load matches, giving up\n')
        exit()
```

Finally, the real work of grabbing the score data for each match. See below the Python code for a sample of the HTML that we'll be navigating for each individual match.

In the current version of ESPN's scoreboard, each match is contained in a `div` with class `boxscore-wrapper` that itself contains two table rows with class `linescore`, where the winner's row also has class `winner`. While the mobile version of the site always lists the winner first (in the format 'Winner d. Loser \<score\>), the full version may show either the winner or loser first, so we have to determine which in order to assign the names and scorelines to the appropriate columns in the dataframe.

Seeds are listed in the same table element as player names, but are captured in their own `span` tags, which makes them easy to extract; then, we can use a regular expression to grab the name itself.

The set scores for each player follow the player name as sibling `td` elements—not as intuitive to scrape as the typical inline format (e.g. "6-4, 6-3") but it's easy enough to grab the scores for each player with BeautifulSoup's `next_siblings` method and put them back together into a single string. Tiebreak scores are listed as superscripts, but can also just be picked up by looking for set scores that are more than one character long. Since only the loser's score is typically listed for a tiebreak, we can convert to integers and pick the `min` value to figure out which one to display.

It's worth noting that, depending on our data processing needs, ESPN's boxscore-style format might actually be better. If we wanted to tally games won, for example, the inline scores that this script pieces together would have to be exploded out again. But in addition to making the information human-readable and reducing the number of fields in our table, this format makes for a better exercise in manipulating HTML that looks good on the page but requires some extra code to make sense of programmatically.


```python
matches = soup.find_all('div', class_="boxscore-wrapper")

for m in matches:
    linescores = m.find_all('tr', class_='linescore')
    if 'winner' in linescores[0].get_attribute_list('class'):
        winner, loser = linescores
    else:
        loser, winner = linescores

    winner_and_seed = winner.find('td', class_='player-cell')
    winner_name = re.sub("\([0-9]+\) ", "", winner_and_seed.text)
    winner_seed = winner.find('span', class_='seed')
    if winner_seed != None:
        winner_seed = int(winner_seed.text.strip('( )'))
    else:
        winner_seed = "None"


    loser_and_seed = loser.find('td', class_='player-cell')
    loser_name = re.sub("\([0-9]+\) ", "", loser_and_seed.text)
    loser_seed = loser.find('span', class_='seed')
    if loser_seed != None:
        loser_seed = int(loser_seed.text.strip('( )'))
    else:
        loser_seed = "None"

    num_sets = len(m.find_all('td', class_="set"))
    sets = zip(winner_and_seed.next_siblings, loser_and_seed.next_siblings)
    match_score = []
    for s in sets:
        set_score = s[0].text[0] + '-' + s[1].text[0]
        if len(s[0].text) > 1:
            tb_score = str(min(int(s[0].text[1:]), int(s[1].text[1:])))
            set_score += '(' + tb_score + ')'
        match_score.append(set_score)
    match_score = ' '.join(match_score)

    df = df.append({'date': yest_string,
                    'winner': winner_name,
                    'w_seed': winner_seed,
                    'loser': loser_name,
                    'l_seed':loser_seed,
                    'score': match_score},
                   ignore_index=True)

df.to_excel(data_file)
```

Here's a sample of the HTML for a single match:


```html
<div class="boxscore-wrapper">
    <table class="boxscore">
        <tr class="boxscore-header">
            <td>
             FINAL
            </td>
            <td align="center" class="set">
             1
            </td>
            <td align="center" class="set">
             2
            </td>
        </tr>
        <tr class="linescore winner">
            <td class="player-cell">
                <a href="/general/tennis/playercard?playerId=2860">
                    <img class="flag" onerror="this.style.display='none'" src="http://a.espncdn.com/i/flags/56x34/can.gif"/>
                    <span class="seed">
                     (3)
                    </span>
                 Denis Shapovalov
                </a>
            </td>
            <td align="center">
             6
            </td>
            <td align="center">
             6
            </td>
        </tr>
        <tr class="linescore">
            <td class="player-cell">
                <a href="/general/tennis/playercard?playerId=2726">
                    <img class="flag" onerror="this.style.display='none'" src="http://a.espncdn.com/i/flags/56x34/pol.gif"/>
                    <span class="seed">
                     (13)
                    </span>
                 Hubert Hurkacz
                </a>
            </td>
            <td align="center">
             4
            </td>
            <td align="center">
             3
            </td>
        </tr>
        <tr class="boxscore-footer">
            <td colspan="3">
             Mens Singles - 3rd Round - Centre Court
            </td>
        </tr>
        <tr class="boxscore-link">
            <td colspan="3">
                <a href="courtcast?matchId=106989">
                  Courtcast
                </a>
            </td>
        </tr>
    </table>
</div>
```

### Automating Scraping with Airflow

It's very much overkill to use Airflow to automate a single script like this one—a quick `echo "0 0 * * * python scraper.py" | chrontab` would do the trick. But my goal was to learn a new data pipeline tool, so here we are. Airflow is not particularly difficult to set up, though most of the inital work happens at the command line: installing, creating an admin account, running the webserver and scheduler, etc. Airflow's whole purpose is to run DAGs, or sets of tasks with acyclic dependencies; for the moment, I only have one task so I don't need to worry about dependencies, but from here it would be easy to add more tasks for data processing, warehousing, etc.

Though my task is really a python script, it's easiest to run it as a bash operator, since Airflow's Python operator just runs a single function that has to be defined in the same file or stored in the same directory as the DAG. The interval is set with chron-style notation, in this case running at 4am UTC, which is midnight EDT at the time of writing. This wouldn't be enough of a cushion in some cases, since night matches at North American tournaments can easily run past midnight, but it'll work for now.


```python
import datetime as dt
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'brian miller',
    'start_date': dt.datetime(2021, 3, 19),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=5),
}

with DAG('scraping_test_dag',
         default_args=default_args,
         schedule_interval='0 4 * * *',
    ) as dag:

    scrape_operator = BashOperator(task_id='ESPN_scrape',
                                   bash_command='python /Users/brian_miller/Data_Science/tennis_pipeline/scrape_test.py')

```

In the future, I'm planning to add some addtional steps to this pipeline, including feeding matches into the predictive model from last time, comparing with the actual results, and producing reports and visualizations. Stay tuned!
