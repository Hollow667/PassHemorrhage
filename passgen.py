import configparser
import csv
import functools
import gzip
import os
import sys

CONFIG = {}


def read_config(filename):
    """Read the given configuration file and update global variables to reflect
    changes (CONFIG)."""

    if os.path.isfile(filename):

        config = configparser.ConfigParser()
        config.read(filename)

        CONFIG["global"] = {
            "chars": config.get("specialchars", "chars").split(","),
            "numfrom": config.getint("nums", "from"),
            "numto": config.getint("nums", "to"),
            "wcfrom": config.getint("nums", "wcfrom"),
            "wcto": config.getint("nums", "wcto"),
            "threshold": config.getint("nums", "threshold"),
        }

        leet = functools.partial(config.get, "leet")
        leetc = {}
        letters = {"a", "i", "e", "t", "o", "s", "g", "z"}

        for letter in letters:
            leetc[letter] = config.get("leet", letter)

        CONFIG["LEET"] = leetc

        return True

    else:
        print("Configuration file " + filename + " not found!")
        sys.exit("Exiting.")

        return False


def leet(x):
    """convert string to leet"""
    for letter, leetletter in CONFIG["LEET"].items():
        x = x.replace(letter, leetletter)
    return x


def concatinations(seq, start, stop):
    for mystr in seq:
        for num in range(start, stop):
            yield mystr + str(num)


def combo(seq, start, special=""):
    for mystr in seq:
        for mystr1 in start:
            yield mystr + special + mystr1

def generate_wordlist(profile):
    """ Generates a wordlist from a given profile """

    chars = CONFIG["global"]["chars"]
    numfrom = CONFIG["global"]["numfrom"]
    numto = CONFIG["global"]["numto"]

    profile["spechars"] = []

    if profile["spechars1"] == "y":
        for spec1 in chars:
            profile["spechars"].append(spec1)
            for spec2 in chars:
                profile["spechars"].append(spec1 + spec2)
                for spec3 in chars:
                    profile["spechars"].append(spec1 + spec2 + spec3)

    print("\r\n[+] Now making a dictionary...")

    words = []
    words = list(map(str.title, profile["words"]))

    word = profile["words"] + words

    combinations = {}
    combinations[1] = list(combo(word, word))
    combinations[1] += list(combo(word, "_"))
    combinations[2] = [""]
    if profile["randnum"] == "y":
        combinations[2] = list(concatinations(word, numfrom, numto))

    combinations001 = [""]

    if len(profile["spechars"]) > 0:
        combinations001 = list(combo(word, profile["spechars"]))

    print("[+] Sorting list and removing duplicates...")

    comb_unique = {}
    for i in range(1, 2):
        comb_unique[i] = list(dict.fromkeys(combinations[i]).keys())

    comb_unique1 = list(dict.fromkeys(word).keys())
    comb_unique2 = list(dict.fromkeys(combinations001).keys())

    uniqlist = comb_unique1

    for i in range(1, 2):
        uniqlist += comb_unique[i]

    uniqlist += comb_unique2

    unique_list1 = list(dict.fromkeys(uniqlist).keys())
    unique_leet = []
    if profile["leetmode"] == "y":
        for x in unique_list1:
            x = leet(x)
            unique_leet.append(x)

    unique_list = unique_list1 + unique_leet

    generate_wordlist.unique_list_finished = []
    generate_wordlist.unique_list_finished = [x for x in unique_list if len(x) < CONFIG["global"]["wcto"] and len(x) > CONFIG["global"]["wcfrom"]]

    return generate_wordlist.unique_list_finished

def main():

    read_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), "passgen.cfg"))

    print("\r\n[+] Insert the information about the victim to make a dictionary")

    profile = {}

    profile["words"] = [""]
    words1 = ""
    words1 = input("> Please enter the words, separated by ';' , spaces will be removed: ").replace(" ", "")
    profile["words"] = words1.split(";")

    profile["spechars1"] = input("> Do you want to add special chars at the end of words? Y/[N]: ").lower()

    profile["randnum"] = input("> Do you want to add some random numbers at the end of words? Y/[N]:").lower()
    profile["leetmode"] = input("> Leet mode? (i.e. leet = 1337) Y/[N]: ").lower()

    generate_wordlist(profile)
