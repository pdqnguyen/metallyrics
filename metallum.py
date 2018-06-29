from utils import scrape_html

def get_band_info(url):
    soup = scrape_html(url)
    name = soup.find('h1', attrs={'class': 'band_name'}).text
    raw_info = soup.find('div', attrs={'id': 'band_stats'})
    keys = raw_info.find_all('dt')
    vals = raw_info.find_all('dd')
    info = {key.text.replace(':', ''): val.text for key, val in zip(*(keys, vals))}
    return info