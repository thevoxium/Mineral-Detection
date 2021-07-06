import requests
import click


@click.command()
@click.option('--path', default = "test.jpg", help = "Provide the path for image.")
def get_result(path):
    resp = requests.post("https://min-dec.herokuapp.com/predict",
        files={"file": open(path,'rb')}
    )
    print(resp.json())

if __name__ == '__main__':
    get_result()
