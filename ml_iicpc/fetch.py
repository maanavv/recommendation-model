import os.path
from scipy.sparse import coo_matrix


def no_file():
    print("Dataset not found, please download the file from:")
    print("http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz (1.6GB)")
    print("then place it in recommender_system_py/data/ and change the file_path")


def fetch_lastfm(min_plays=200):
    file_path = "data/lastfm.tsv"

    if not os.path.exists(file_path):
        no_file()
        

    data, row, col = [], [], []
    artists, users = {}, {}

    with open(file_path, encoding="utf-8") as data_file:
        for line in data_file:
            readable_data = line.strip().split("\t")

            user = readable_data[0]
            artist_id = readable_data[1]
            artist_name = readable_data[2]
            plays = int(readable_data[3])

            if user not in users:
                users[user] = len(users)

            if artist_id not in artists:
                artists[artist_id] = {
                    "name": artist_name,
                    "id": len(artists),
                }

            if plays > min_plays:
                data.append(plays)
                row.append(users[user])
                col.append(artists[artist_id]["id"])

    coo = coo_matrix((data, (row, col)))

    return {
        "matrix": coo,
        "artists": artists,
        "users": len(users),
    }
