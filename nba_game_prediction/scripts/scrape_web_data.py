import pandas as pd
import requests
from bs4 import BeautifulSoup

# # TODO
# def scrape_odd_maker_data(url, season):
#     driver = webdriver.Firefox()
#     driver.get(url)
#     driver.page_source
#     # res = driver.find_element(By.NAME, 'xeid="ULD28jPK"')
#     # print(res)
#     res = driver.find_element(By.ID, "tournamentTable")
#     print(res.get_attribute("innerHTML"))
#     # __import__('pdb').set_trace()
#     driver.close()
#     salary_table = soup.find("table")
#     length = len(salary_table.find_all("td"))
#     print(salary_table)
#     print(length)
#     __import__("pdb").set_trace()


def scrape_team_salaries(url, season):
    r = requests.get(url)
    r_html = r.text
    soup = BeautifulSoup(r_html, "html.parser")
    salary_table = soup.find("table")
    length = len(salary_table.find_all("td"))
    entries_per_row = 4
    team_names = [
        salary_table.find_all("td")[i].text.strip()
        for i in range(5, length, entries_per_row)
    ]
    salaries = [
        salary_table.find_all("td")[i].text.strip()
        for i in range(6, length, entries_per_row)
    ]
    inflation_adjusted_salaries = [
        salary_table.find_all("td")[i].text.strip()
        for i in range(7, length, entries_per_row)
    ]
    df_dict = {
        "player_names": team_names,
        "salary": salaries,
        "inflation adjusted salary": inflation_adjusted_salaries,
        "season": season,
    }

    salary_df = pd.DataFrame(df_dict)
    print(len(salary_df))
    print(salary_df.head())


def main():
    seasons = [
        "2017-2018",
        # "2018-2019"
    ]
    for season in seasons:
        # TODO write them to SQL db
        scrape_team_salaries(f"https://hoopshype.com/salaries/{season}/", season)
        # scrape_odd_maker_data(
        # f"https://www.oddsportal.com/basketball/usa/nba-{season}/results/",
        # season)


if __name__ == "__main__":
    main()
