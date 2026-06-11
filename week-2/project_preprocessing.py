import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


# -----------------------------------------------------------
# Preprocessing helper function
# -----------------------------------------------------------


def clean_before_preprocessing(X_train, X_test, leakage_cols):
    X_train = X_train.copy()
    X_test = X_test.copy()

    cols_to_drop = ["Misc Feature", "Misc Val"] + leakage_cols

    X_train = X_train.drop(columns=cols_to_drop, errors="ignore")
    X_test = X_test.drop(columns=cols_to_drop, errors="ignore")

    # Manual correction: this value is most likely a typo, not a systemic pattern
    # See EDA: unrealistic values check
    X_train["Garage Yr Blt"] = X_train["Garage Yr Blt"].replace(2207, 2007)
    X_test["Garage Yr Blt"] = X_test["Garage Yr Blt"].replace(2207, 2007)

    return X_train, X_test


# -----------------------------------------------------------
# Structural categorical nominal columns
# -----------------------------------------------------------

# temp solution - just dropped 12 rows that have unexplained by structural missingness missing valuues

# there should be some ordinal and some norminal columns

structural_cat_nominal_cols = [
    "Mas Vnr Type", # тип облицовочной кладки # 7 unexplained
    "Garage Type" # тип гаража - пристройка, в здании и так далее
]


# -----------------------------------------------------------
# Structural categorical ordinal columns
# -----------------------------------------------------------

structural_cat_ordinal_cols = [
    "Bsmt Exposure", # насколько подвал выходит наружу
    "BsmtFin Type 2", # качество отделки подвала ком 2
    "Garage Finish", # степень отделки гаража
    "Garage Qual", # качества гаража
    "Garage Cond", # состояние гаража
    "Bsmt Qual", # качество подваал
    "Bsmt Cond", # состояние подвала
    "BsmtFin Type 1", # качество отделки подвала ком 2
    "Fireplace Qu", # качество камина
    "Pool QC", # качество бассейна
    "Fence", # типы забора, ординальный ли?
    "Alley"#типы аллеи, ординальный ли?
]


# -----------------------------------------------------------
# Structural categorical ordinal category order
# -----------------------------------------------------------

# вручную
structural_cat_ordinal_categories = [
    # Bsmt Exposure
    ["None", "No", "Mn", "Av", "Gd"],

    # BsmtFin Type 2
    ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],

    # Garage Finish
    ["None", "Unf", "RFn", "Fin"],

    # Garage Qual
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],

    # Garage Cond
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],

    # Bsmt Qual
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],

    # Bsmt Cond
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],

    # BsmtFin Type 1
    ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],

    # Fireplace Qu
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],

    # Pool QC
    ["None", "Fa", "TA", "Gd", "Ex"],

    # Fence
    ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],

    # Alley
    ["None", "Grvl", "Pave"]
]


# -----------------------------------------------------------
# Structural numerical columns
# -----------------------------------------------------------

# temp solution - just dropped 12 rows that have unexplained by structural missingness missing valuues

structural_num_cols = [
    "Garage Yr Blt", 
    "BsmtFin SF 1", 
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Garage Cars",
    "Garage Area",
    "Mas Vnr Area"
]


# -----------------------------------------------------------
# Full categorical nominal columns
# -----------------------------------------------------------

full_cat_nominal_cols = [
    "MS SubClass",
    "MS Zoning",
    "Street",
    "Land Contour",
    "Lot Config",
    "Neighborhood",
    "Condition 1",
    "Condition 2",
    "Bldg Type",
    "House Style",
    "Roof Style",
    "Roof Matl",
    "Exterior 1st",
    "Exterior 2nd",
    "Foundation",
    "Heating",
    "Central Air"
]


# -----------------------------------------------------------
# Full categorical ordinal columns
# -----------------------------------------------------------

full_cat_ordinal_cols = [
    "Lot Shape", # форма участка
    "Utilities", # доступные коммунальные услуги
    "Land Slope", # уклон участка
    "Overall Qual", 
    "Overall Cond", 
    "Exter Qual", 
    "Exter Cond", 
    "Heating QC", # качество и состояние отопления
    "Kitchen Qual", #
    "Functional", # используемость дома
    "Paved Drive" # тип подъездной дорожки
]


# -----------------------------------------------------------
# Full categorical ordinal category order
# -----------------------------------------------------------

full_cat_ordinal_categories = [
    # Lot Shape
    ["Reg", "IR1", "IR2", "IR3"],

    # Utilities
    ["ELO", "NoSeWa", "NoSewr", "AllPub"],

    # Land Slope
    ["Gtl", "Mod", "Sev"],

    # Overall Qual
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    # Overall Cond
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    # Exter Qual
    ["Po", "Fa", "TA", "Gd", "Ex"],

    # Exter Cond
    ["Po", "Fa", "TA", "Gd", "Ex"],

    # Heating QC
    ["Po", "Fa", "TA", "Gd", "Ex"],

    # Kitchen Qual
    ["Po", "Fa", "TA", "Gd", "Ex"],

    # Functional
    ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],

    # Paved Drive
    ["N", "P", "Y"]
]


# -----------------------------------------------------------
# Full numerical columns
# -----------------------------------------------------------

full_num_cols = [
    "Lot Area",
    "Year Built",
    "Year Remod/Add",
    "1st Flr SF",
    "2nd Flr SF",
    "Low Qual Fin SF",
    "Gr Liv Area",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "Kitchen AbvGr",
    "TotRms AbvGrd",
    "Fireplaces",
    "Wood Deck SF",
    "Open Porch SF",
    "Enclosed Porch",
    "3Ssn Porch",
    "Screen Porch",
    "Pool Area"
]