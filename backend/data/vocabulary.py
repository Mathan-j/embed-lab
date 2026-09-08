"""The 600-word vocabulary the whole app is built on: 6 categories x 100 words.

Single lowercase words only (no spaces, no hyphens) so every word is one clean
`text=` query param and one row of the embedding matrix. Each category mixes short,
common words that the tokenizer keeps as one wordpiece (`cat`, `red`, `chef`) with
longer or rarer ones that split into subwords (`chinchilla`, `heliotrope`,
`anthropologist`) — the tokenize card has nothing to show if every word is one
piece, so roughly a third are picked to split. Nobody in this dict was fetched from
a list; they were chosen by hand for these two properties plus one more: no word
appears in two categories, which `tests/test_vocab.py` enforces.
"""

VOCABULARY: dict[str, list[str]] = {
    "animal": [
        "cat", "dog", "cow", "pig", "hen", "fox", "owl", "bat", "ant", "bee",
        "rat", "elk", "ram", "yak", "ape", "cod", "eel", "jay", "pug", "koi",
        "horse", "sheep", "goat", "mouse", "camel", "zebra", "tiger", "lion", "bear", "wolf",
        "deer", "moose", "otter", "mole", "hare", "rabbit", "ferret", "weasel", "beaver", "badger",
        "raccoon", "opossum", "squirrel", "chipmunk", "hedgehog", "tortoise", "iguana", "gecko", "chameleon", "crocodile",
        "alligator", "cobra", "viper", "anaconda", "piranha", "barracuda", "stingray", "jellyfish", "octopus", "squid",
        "lobster", "crab", "shrimp", "oyster", "mussel", "snail", "slug", "earthworm", "ladybug", "dragonfly",
        "butterfly", "moth", "cricket", "beetle", "termite", "cockroach", "mosquito", "wasp", "hornet", "spider",
        "centipede", "millipede", "eagle", "falcon", "hawk", "vulture", "sparrow", "robin", "pigeon", "seagull",
        "pelican", "flamingo", "peacock", "ostrich", "penguin", "dolphin", "walrus", "buffalo", "antelope", "gazelle",
    ],
    "food": [
        "bread", "cheese", "rice", "soup", "pepper", "noodle", "pasta", "pizza", "burger", "sandwich",
        "salad", "sausage", "bacon", "omelet", "pancake", "waffle", "muffin", "biscuit", "cracker", "pretzel",
        "popcorn", "peanut", "almond", "walnut", "cashew", "pistachio", "hazelnut", "coconut", "banana", "apple",
        "grape", "cherry", "peach", "apricot", "papaya", "mango", "pineapple", "watermelon", "cantaloupe", "strawberry",
        "blueberry", "raspberry", "blackberry", "cranberry", "pomegranate", "grapefruit", "lemon", "lime", "kiwi", "fig",
        "raisin", "prune", "lettuce", "cabbage", "broccoli", "cauliflower", "spinach", "kale", "celery", "cucumber",
        "zucchini", "eggplant", "pumpkin", "squash", "carrot", "potato", "yam", "onion", "garlic", "mushroom",
        "tomato", "radish", "beet", "artichoke", "asparagus", "cornbread", "tortilla", "burrito", "taco", "enchilada",
        "quesadilla", "dumpling", "sushi", "ramen", "curry", "stew", "chowder", "casserole", "lasagna", "risotto",
        "paella", "falafel", "hummus", "yogurt", "custard", "pudding", "caramel", "toffee", "fudge", "marshmallow",
    ],
    "vehicle": [
        "car", "truck", "bicycle", "tram", "ferry", "glider", "scooter", "van", "bus", "taxi",
        "jeep", "tractor", "motorcycle", "moped", "segway", "unicycle", "tricycle", "skateboard", "wagon", "cart",
        "sled", "sleigh", "canoe", "kayak", "rowboat", "sailboat", "yacht", "submarine", "schooner", "catamaran",
        "hovercraft", "speedboat", "tugboat", "freighter", "tanker", "cruiser", "destroyer", "battleship", "airplane", "biplane",
        "helicopter", "seaplane", "jetliner", "spaceship", "rocket", "shuttle", "blimp", "zeppelin", "balloon", "gondola",
        "funicular", "monorail", "subway", "streetcar", "locomotive", "caboose", "boxcar", "forklift", "bulldozer", "excavator",
        "crane", "dumptruck", "backhoe", "snowplow", "snowmobile", "ambulance", "firetruck", "hearse", "limousine", "convertible",
        "sedan", "hatchback", "minivan", "pickup", "camper", "caravan", "trailer", "rickshaw", "chariot", "stagecoach",
        "carriage", "buggy", "wheelbarrow", "handcart", "minibus", "golfcart", "dunebuggy", "gokart", "bobsled", "toboggan",
        "hangglider", "paraglider", "jetski", "hydrofoil", "minisub", "dinghy", "raft", "barge", "longboat", "riverboat",
    ],
    "emotion": [
        "joy", "dread", "envy", "relief", "grief", "delight", "unease", "fear", "anger", "rage",
        "wrath", "hate", "love", "lust", "pride", "shame", "guilt", "hope", "despair", "sorrow",
        "sadness", "happiness", "gladness", "cheerfulness", "excitement", "enthusiasm", "eagerness", "anticipation", "nostalgia", "melancholy",
        "loneliness", "boredom", "apathy", "indifference", "contentment", "satisfaction", "gratitude", "appreciation", "admiration", "awe",
        "wonder", "curiosity", "confusion", "bewilderment", "frustration", "irritation", "annoyance", "agitation", "anxiety", "worry",
        "nervousness", "tension", "stress", "panic", "terror", "horror", "disgust", "contempt", "scorn", "jealousy",
        "resentment", "bitterness", "regret", "remorse", "embarrassment", "humiliation", "mortification", "awkwardness", "discomfort", "calmness",
        "serenity", "tranquility", "peace", "harmony", "elation", "euphoria", "bliss", "ecstasy", "rapture", "exhilaration",
        "thrill", "exuberance", "jubilation", "triumph", "vindication", "defeat", "disappointment", "dejection", "gloom", "despondency",
        "hopelessness", "helplessness", "vulnerability", "insecurity", "timidity", "shyness", "longing", "yearning", "homesickness", "sympathy",
    ],
    "colour": [
        "crimson", "teal", "ochre", "indigo", "beige", "magenta", "scarlet", "maroon", "burgundy", "vermilion",
        "azure", "cerulean", "cobalt", "navy", "turquoise", "cyan", "aquamarine", "emerald", "jade", "verdigris",
        "chartreuse", "forest", "sage", "lavender", "lilac", "violet", "purple", "mauve", "periwinkle", "amber",
        "gold", "bronze", "copper", "rust", "sienna", "umber", "tan", "khaki", "ivory", "cream",
        "pearl", "silver", "gray", "grey", "charcoal", "slate", "ebony", "jet", "onyx", "obsidian",
        "black", "white", "brown", "chestnut", "mahogany", "hazel", "coral", "rose", "blush", "fuchsia",
        "pink", "red", "yellow", "green", "blue", "cerise", "wine", "ruby", "garnet", "topaz",
        "sapphire", "amethyst", "citrine", "opal", "pewter", "platinum", "brass", "taupe", "fawn", "buff",
        "flax", "wheat", "sand", "sandstone", "terracotta", "tangerine", "mustard", "saffron", "marigold", "canary",
        "auburn", "russet", "puce", "ecru", "vermeil", "celadon", "cinnabar", "malachite", "heliotrope", "gunmetal",
    ],
    "job": [
        "plumber", "surgeon", "teacher", "welder", "chemist", "baker", "butcher", "carpenter", "electrician", "mechanic",
        "farmer", "fisherman", "librarian", "journalist", "editor", "photographer", "illustrator", "architect", "engineer", "scientist",
        "biologist", "physicist", "geologist", "astronomer", "pharmacist", "dentist", "therapist", "psychologist", "psychiatrist", "nurse",
        "paramedic", "firefighter", "detective", "lawyer", "judge", "accountant", "auditor", "economist", "banker", "cashier",
        "waiter", "waitress", "bartender", "chef", "cook", "barista", "hairdresser", "barber", "tailor", "seamstress",
        "cobbler", "locksmith", "blacksmith", "goldsmith", "jeweler", "watchmaker", "upholsterer", "roofer", "bricklayer", "mason",
        "plasterer", "painter", "decorator", "gardener", "landscaper", "florist", "beekeeper", "shepherd", "rancher", "veterinarian",
        "zookeeper", "lifeguard", "referee", "umpire", "coach", "athlete", "gymnast", "dancer", "choreographer", "musician",
        "composer", "conductor", "singer", "actor", "actress", "director", "producer", "screenwriter", "novelist", "poet",
        "translator", "interpreter", "proofreader", "cartographer", "surveyor", "archaeologist", "anthropologist", "historian", "curator", "archivist",
    ],
}

# Words that genuinely belong to two of the six categories above (colour/food,
# animal/food, animal/vehicle, ...). Excluded from VOCABULARY and from training:
# the classifier will confidently pick one category for these, and that is not a
# bug, it is the point -- the label is underdetermined, not the model.
# "olive" joined this list (moved out of `colour`) because it is exactly the same
# shape as "orange"/"mint": a food that is also a colour name. `colour` was
# backfilled with "verdigris" to keep the category at 100.
AMBIGUOUS: list[str] = [
    "orange", "mint", "turkey", "jaguar", "plum", "salmon", "ginger", "date", "olive",
]

# Pairs (plain_word, compound_word) where the compound tokenizes with the plain
# word as one of its wordpieces -- e.g. "catboat" -> ["cat", "##boat"] -- which
# pulls its embedding toward the plain word even though the two mean unrelated
# things. This is a real property of subword tokenization, not a bug: measured
# with this model, cos(cat, catboat) = 0.5941, higher than cos(cat, tiger) =
# 0.5453, a genuine animal neighbour. But the model still gets the *meaning*
# right -- cos(catboat, boat) = 0.7455 far exceeds the "cat" leakage, and
# cos(catboat, dog) = 0.3205 is low -- so the lesson is that embeddings are
# *mostly* semantic, with a measurable subword-driven wobble, not that they are
# purely spelling-driven the way cos(cat, car) > cos(cat, dog) would be.
#
# Excluded from VOCABULARY and from training for the same reason AMBIGUOUS is:
# the compound word is not a clean example of its category and would just add
# noise to the classifier and the neighbour list.
#
# Verified by direct measurement before inclusion (see the numbers above and
# below); several plausible-looking pairs were tried and dropped because they
# did NOT show the "meaning wins, subword only wobbles it" shape -- for example
# cos(bee, beetroot) = 0.7689 is *higher* than cos(beetroot, beet) = 0.6931, so
# beetroot's embedding is dominated by the shared "bee" token rather than
# recovering its true meaning, and "car"/"carpet" does not even tokenize with a
# shared piece ("carpet" is one wordpiece, not "car" + "##pet"). Both were
# dropped rather than included to force the phenomenon.
#
# cos(ram, ramrod) = 0.4425, above cos(ram, goat) = 0.2633 (a genuine animal
# neighbour), while cos(ramrod, rod) = 0.5734 exceeds the "ram" leakage and
# cos(ramrod, cheese) = 0.2282 is low -- the same shape as cat/catboat.
SUBWORD_LEAKAGE: list[tuple[str, str]] = [
    ("cat", "catboat"),
    ("ram", "ramrod"),
]
