import numpy as np
from lightfm import LightFM
from fetch import fetch_lastfm


def get_recommendations(model, coo_mtrx, data, user_ids):
    n_users, n_items = coo_mtrx.shape

    for user in user_ids:
        user = int(user)

        if user < 0 or user >= n_users:
            print(f"User {user} out of range, skipping")
            continue

        scores = model.predict(user, np.arange(n_items))
        top_scores = np.argsort(-scores)[:3]

        print(f"Recommendations for user {user}:")

        for x in top_scores:
            for values in data["artists"].values():
                if x == values["id"]:
                    print(f"  - {values['name']}")

        print()


if __name__ == "__main__":
    print("Loading data...")
    data = fetch_lastfm()

    print("Training LightFM model...")
    model = LightFM(loss="warp")
    model.fit(data["matrix"], epochs=30, num_threads=2)

    # SAFE users
    users = [0, 1, 2]

    print("Generating recommendations...")
    get_recommendations(model, data["matrix"], data, users)

    print("Recommender system setup complete.")
