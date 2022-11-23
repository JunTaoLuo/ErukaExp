import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Update sample labels",
        description="Script for uploading labeling results from csv file")
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output, including executed SQL queries')
    parser.add_argument('entries', metavar="N", type=int, help="Number of entries to retrieve", default=20, nargs='?')
    args = parser.parse_args()

