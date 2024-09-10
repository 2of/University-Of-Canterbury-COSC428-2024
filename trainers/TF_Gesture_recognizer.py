import tensorflow as tf
import pandas as pd
import numpy as np
# Step 1: Read CSV files
pinch_data = pd.read_csv("pinch_one_hand_cleaned.csv", header=None, names=[i for i in range(1, 121)] + ['outcome'])
point_data = pd.read_csv("point_cleaned.csv", header=None, names=[i for i in range(1, 121)] + ['outcome'])
SINGULAR_POINT = [0.021822521216514547,0.029557978459252968,0.05456009575093399,0.08287292022636437,0.12015012362106932,0.18866710972649445,0.23175666895163954,0.26687031384797716,0.15472365483101766,0.10976496043703203,0.09174387200788692,0.08503985140095179,0.15017360389883375,0.10175585127840234,0.07670379625951068,0.021052066606624238,0.05355284433190241,0.08284871746403345,0.13452043432857191,0.20036983663214092,0.24207082011052453,0.27612133367998815,0.16843781247680076,0.10988088908563727,0.07923253826049154,0.07010387323528891,0.16205271351861136,0.09764234603775412,0.06367075584338053,0.03270179650104364,0.06188893775040715,0.1196792552902043,0.18289994342362423,0.22362568800337979,0.25706868299667796,0.15265032325944214,0.08886770649114716,0.06228593782296812,0.05560079016626253,0.14493825736941368,0.07681934247913903,0.04715519537239,0.029350742898971553,0.09624106515759424,0.15403305789817318,0.1931953233883733,0.225805505765494,0.12677385270946748,0.05647544147701674,0.04936499650491394,0.05066927854239403,0.11700153694031505,0.047329739241111224,0.039498556527283235,0.08490298948456013,0.132533550290007,0.16885934789125898,0.19997274336587384,0.11035467580963873,0.0271477041683836,0.05048100522158373,0.05904534211279699,0.09784210433470258,0.024073396560735292,0.04949001280195144,0.07140888199029004,0.11646442439684697,0.15333531997057898,0.034759711766318424,0.084240174272057,0.13524898684608036,0.1424105078174516,0.03514043510981733,0.10250008474631453,0.13184156139429573,0.04524705562516701,0.0823876496548276,0.03857002327220928,0.11706352068893607,0.17986933739727928,0.19067160402423283,0.038495339302932974,0.13976002738160717,0.18183089819583328,0.03721145511655274,0.08371272149980358,0.14905192511536974,0.21249784866839883,0.22455298003642765,0.08244917128968797,0.1714673860657353,0.21680053279642045,0.12090634324350154,0.1779641578412559,0.2405258684425257,0.25327299662600966,0.11888818791766774,0.19964197183133095,0.24635854493366538,0.1020415806862143,0.16042836971177443,0.16934857040425388,0.017306426284385872,0.12336684627761546,0.159412406349919,0.06351973976464297,0.07555848696718855,0.0868953734374018,0.022734940205694176,0.06840729348370536,0.014070300508092737,0.1471222619262885,0.04104936922196139,0.015569549168077186,0.15681268146818472,0.05365670052812427,0.01117570526337362,0.10881355475690019,0.147300749985344,0.04766264691671676]
SINGULAR_PINCH = [0.04305551353639492,0.04935061502586901,0.0719246242222137,0.11336693443823265,0.11870184875119698,0.12959078624373704,0.1361061467544862,0.14434995291974453,0.1485006179961358,0.06922993819781009,0.040447099260488464,0.0397785826825343,0.15066529505093917,0.07898208866177717,0.04222385130471481,0.018956922904516196,0.049569905943643376,0.08910668537764194,0.16157137169939909,0.17255949407076968,0.17858175096902573,0.185833273008918,0.19145016211050203,0.10463991736390842,0.06472003578317072,0.07621668632158898,0.19250003422070128,0.10599759455950614,0.06330946646427288,0.030681701513157803,0.07104046674321018,0.16666637291778874,0.175117997393132,0.17944858008993003,0.1848631263052314,0.19378141565384693,0.0999380804126067,0.05764806184541576,0.07385524838246663,0.19201678466819133,0.09681338885213964,0.05507952899671054,0.04184647325151809,0.181722681110608,0.186396356009108,0.18812088061087096,0.19058936742543933,0.2043634437758608,0.10299971227627085,0.06181281105400386,0.08301278846971545,0.19835790227301492,0.09207057121988024,0.057734795379228106,0.21806013233958094,0.21988529352791616,0.21943754861393522,0.21922108094472217,0.23698917427503288,0.1328209354742431,0.0963329268479447,0.11859780854640528,0.22738639271735106,0.11555833965971786,0.09187514046520909,0.024945514048605226,0.041317819646730106,0.05977512622135677,0.03880507489995298,0.09112121380838507,0.12172993888553303,0.09949808203390394,0.05949524635367708,0.11609426215153751,0.1261850121323091,0.016557798732097305,0.03520480972013534,0.018930381560532783,0.08801457132207545,0.1247754048979011,0.10345942013424515,0.034551348549599764,0.11060513978669881,0.12904313205818208,0.0186647092326453,0.019757076258136563,0.08661661544937618,0.12633428419149087,0.10617293676698508,0.018666658506788104,0.1067765289426607,0.13038830386610434,0.03353243632926356,0.08762271212096998,0.1294547794964957,0.11100816566709382,0.008434316977180115,0.10425701914090184,0.13319571980138575,0.1044072528310574,0.14260111962829244,0.12162323461699713,0.02843232223119201,0.12564052316376162,0.14679810148116743,0.04265858504466359,0.029530748139589613,0.09548278430825975,0.025869118087438976,0.04602600958518962,0.022295841910769637,0.1370390079041754,0.041606878697776545,0.0044838735433119226,0.1181896249117109,0.04229940108422237,0.026775604904426552,0.11256487974802745,0.14084507588473996,0.04268902891863132]

combined_data = pd.concat([pinch_data, point_data], ignore_index=True)




def preprocess_data(df):
    if 'outcome' in df.columns:
        X = df.drop(columns=['outcome']) 
        y = df['outcome'] 
        return X, y
    else:
        return df, None  # If 'outcome' column is not present, return only X


X_pinch, y_pinch = preprocess_data(pinch_data)
X_point, y_point = preprocess_data(point_data)


print("Missing values in X_point:")
print(X_point.isnull().sum())

print("Missing values in y_point:")
print(y_point.isnull().sum())


def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

# Step 4: Train the model on the pinch_data
model_pinch = create_model(input_shape=(X_pinch.shape[1],))
model_pinch.fit(X_pinch, y_pinch, epochs=10)

# Step 5: Train the model on the point_data
model_point = create_model(input_shape=(X_point.shape[1],))
model_point.fit(X_point, y_point, epochs=10)



# Note, this is a row from point_cleaned so should be close to 100% prob
new_data = np.array([SINGULAR_POINT])
probability = model_pinch.predict(new_data)
print(probability)

print("_________")
new_data = np.array([SINGULAR_PINCH])
probability = model_pinch.predict(new_data)
print(probability)
