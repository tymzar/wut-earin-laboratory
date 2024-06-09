import argparse

from pipeline import Pipeline


def main(popularity, temperature):

    recommendation_pipeline = Pipeline(popularity, temperature)

    while True:

        song_name = input("Enter song name: ")

        if (
            input(
                f"Do you want to change popularity or temperature? {popularity=}, {temperature=}: "
            )
            == "y"
        ):
            if input("Do you want to change the popularity? (y/n): ") == "y":
                popularity = int(input("Enter popularity: "))

            if input("Do you want to change the temperature? (y/n): ") == "y":
                temperature = float(input("Enter temperature: "))

        recommendation_pipeline = Pipeline(popularity, temperature)
        recommendation_pipeline.recommend(song_name)

        if input("Do you want to continue? (y/n): ") == "n":
            break


def temperature_type(string):
    value = float(string)
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError(
            f"Temperature must be between 0 and 1, got {value}"
        )
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Song recommendation system")

    parser.add_argument(
        "--popularity",
        type=int,
        default=1,
        help="Minimum popularity of the song to be considered",
    )

    parser.add_argument(
        "--temperature",
        type=temperature_type,
        # max 1 lowest 0
        default=0.5,
        help="Temperature of the recommendation system",
    )

    # python main.py "Shape of You" --popularity 20

    arguments = parser.parse_args()

    main(arguments.popularity, arguments.temperature)
