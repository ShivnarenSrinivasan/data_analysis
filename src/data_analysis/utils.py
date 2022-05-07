import requests


def download(url: str, filename: str = 'helpers.py') -> None:
    """Download file from web.

    Saved to specified path.
    """
    res = requests.get(url)
    with open(filename, 'w') as fp:
        fp.write(res.text)
        print(f'Saved `{filename}`!')
