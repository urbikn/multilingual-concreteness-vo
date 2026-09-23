en_prompt = """Background: A concrete expression refers to an event with which you can have an immediate experience through your senses (smelling, tasting, touching, hearing, seeing) and by performing or witnessing the event. An example of a concrete expression is 'drive car'.
An abstract expression refers to an event or process you cannot experience directly through your senses or by performing the event. An example of an abstract expression is 'deliver justice'.

Task: Which of the four expressions is likely the MOST concrete and which of the four expressions is likely the MOST abstract.

Only provide the expression number. Do not repeat the text content.
"""

de_prompt = """Hintergrund: Ein konkreter Ausdruck bezieht sich auf ein Ereignis, das Sie mit Ihren Sinnen (Riechen, Schmecken, Tasten, Hören, Sehen) und durch Ausführen oder Miterleben des Ereignisses unmittelbar erfahren können. Ein Beispiel für einen konkreten Ausdruck ist „Auto fahren“.
Ein abstrakter Ausdruck bezieht sich auf ein Ereignis oder einen Vorgang, den Sie nicht direkt mit Ihren Sinnen oder durch Ausführen des Ereignisses erfahren können. Ein Beispiel für einen abstrakten Ausdruck ist „Recht sprechen“.

Aufgabe: Welche der vier Ausdrücke ist vermutlich der KONKRETESTE und welche der vier Ausdrücke ist vermutlich der ABSTRAKTESTE?

Geben Sie nur die Nummer des Ausdrucks an. Wiederholen Sie nicht den Textinhalt.
"""

sl_prompt = """Ozadje: Konkretni izraz se nanaša na dogodek, ki ga lahko neposredno doživite s svojimi čutili (vonj, okus, dotik, sluh, vid) in z izvedbo ali pričo dogodka. Primer konkretnega izraza je „voziti avto“.
Abstraktni izraz se nanaša na dogodek ali proces, ki ga ne morete neposredno doživeti s svojimi čutili ali z izvedbo dogodka. Primer abstraktnega izraza je „zagotoviti pravico“.

Naloga: Kateri od štirih izrazov je NAJBOLJ konkreten in kateri NAJBOLJ abstrakten?

Navedite samo številko izraza. Ne ponavljajte vsebine besedila.
"""


def en_bws_prompt(targets):
    target_inputs = "\n".join([f"Expression {i + 1}: {targets[i]}" for i in range(4)]) + "\n"

    return (
        en_prompt
        + target_inputs
        + """
Answer only with two numbers separated by a comma:
<most concrete expression number>, <most abstract expression number>
"""
    )


def en_bws_prompt_few(targets):
    target_inputs = "\n".join([f"Expression {i + 1}: {targets[i]}" for i in range(4)])

    few_shot_prompt = """
Here are examples of expressions with most concrete and most abstract expressions:
Expression 1: hold office
Expression 2: write thing
Expression 3: throw ball
Expression 4: undermine principle
3,4

Expression 1: read magazine
Expression 2: affect neck
Expression 3: develop telescope
Expression 4: unlock creativity
1,4

Expression 1: want cookie
Expression 2: defy logic
Expression 3: decide election
Expression 4: serve beer
4,2

"""

    return (
        en_prompt
        + """
Answer only with two numbers separated by a comma:
<most concrete expression number>, <most abstract expression number>
"""
        + few_shot_prompt
        + target_inputs
    )


def de_bws_prompt(targets):
    target_inputs = "\n".join([f"Ausdruck {i + 1}: {targets[i]}" for i in range(4)]) + "\n"

    return (
        de_prompt
        + target_inputs
        + """
Nur mit zwei durch ein Komma getrennten Zahlen antworten:
<Nummer des konkretesten Ausdrucks>, <Nummer des abstraktesten Ausdrucks>"""
    )


def de_bws_prompt_few(targets):
    target_inputs = "\n".join([f"Ausdruck {i + 1}: {targets[i]}" for i in range(4)])

    few_shot_prompt = """
Hier sind Beispiele für Ausdrücke mit den konkretesten und abstraktesten Ausdrücken:
Ausdruck 1: Ansicht ablehnen
Ausdruck 2: Karte akzeptieren
Ausdruck 3: Ball werfen
Ausdruck 4: Grundsatz verstehen
3,4

Ausdruck 1: Bier trinken
Ausdruck 2: Kontakt herstellen
Ausdruck 3: Wahl organisieren
Ausdruck 4: Logik akzeptieren
1,4

Ausdruck 1: Trennung vorschlagen
Ausdruck 2: Wirkung verursachen
Ausdruck 3: Blut untersuchen
Ausdruck 4: Bild tragen
4,2

"""

    return (
        de_prompt
        + """
Nur mit zwei durch ein Komma getrennten Zahlen antworten:
<Nummer des konkretesten Ausdrucks>, <Nummer des abstraktesten Ausdrucks>"""
        + few_shot_prompt
        + target_inputs
    )


def sl_bws_prompt(targets):
    target_inputs = "\n".join([f"Izraz {i + 1}: {targets[i]}" for i in range(4)]) + "\n"

    return (
        sl_prompt
        + target_inputs
        + """
Odgovorite samo z dvema številkama, ločenima z vejico:
<najbolj konkretna številka izraza>, <najbolj abstraktna številka izraza>"""
    )


def sl_bws_prompt_few(targets):
    target_inputs = "\n".join([f"Izraz {i + 1}: {targets[i]}" for i in range(4)])

    few_shot_prompt = """
Tukaj so primeri izrazov z najbolj konkretnimi in najbolj abstraktnimi izrazi:
Izraz 1: napisati stvar
Izraz 2: opravljati funkcijo
Izraz 3: metati žogo
Izraz 4: spremeniti idejo
3,4

Izraz 1: postreči pivo
Izraz 2: izboljševati sistem
Izraz 3: odločiti volitve
Izraz 4: nasprotovati logiki
1,4

Izraz 1: razviti kri
Izraz 2: vzbuditi radovednost
Izraz 3: izogibati soli
Izraz 4: nositi torbo
4,2

"""

    return (
        sl_prompt
        + """
Odgovorite samo z dvema številkama, ločenima z vejico:
<najbolj konkretna številka izraza>, <najbolj abstraktna številka izraza>"""
        + few_shot_prompt
        + target_inputs
    )


PROMPTS = {
    "en": (en_bws_prompt, en_bws_prompt_few),
    "de": (de_bws_prompt, de_bws_prompt_few),
    "sl": (sl_bws_prompt, sl_bws_prompt_few),
}

corrections_dict_sl = {
    "pripraviti aktivnost": "pripraviti aktivnosti",
    "odreči pravici": "odreči pravico",
    "sklepati posel": "sklepati posle",
    "izpeljati posel": "izpeljati posla",
    "prevzeti posel": "prevzeti posle",
    "spomniti ideje": "spomniti idejo",
    "razložiti sanje": "razložiti sanjo",
    "zaživeti sanje": "zaživeti sanjo",
    "pogrešati informacije": "pogrešati informacijo",
    "izdajati račun": "izdajati račune",
    "zmanjkati baterije": "zmanjkati baterijo",
    "izdajati album": "izdajati albume",
    "ponujati cev": "ponujati cevi",
    "pogrešati dotik": "pogrešati dotike",
    "vključiti pogled": "vključiti poglede",
    "ohraniti mišice": "ohraniti mišico",
    "lotiti knjige": "lotiti knjigo",
    "občutiti znak": "občutiti znake",
    "prevzeti bralca": "prevzeti bralce",
    "raziskovati otok": "raziskovati otoke",
    "potrjevati račun": "potrjevati račune",
    "nadomestiti jajca": "nadomestiti jajce",
    "izdajati dokument": "izdajati dokumente",
    "zdraviti pacienta": "zdraviti paciente",
    "ponujati hotel": "ponujati hotele",
    "odkriti znak": "odkriti znake",
    "pripraviti jajca": "pripraviti jajce",
    "ponujati obrok": "ponujati obroke",
    "povzročati veter": "povzročati vetrove",
    "odkriti recept": "odkriti recepte",
    "preprečevati znak": "preprečevati znake",
    "imeti napoved": "imeti napovedi",
    "lotiti naloge": "lotiti nalogo",
    "pričakovati predlog": "pričakovati predloge",
    "nadaljevati pogovor": "nadaljevati pogovore",
    "spomniti obletnice": "spomniti obletnico",
    "analizirati predlog": "analizirati predloge",
    "lotiti vprašanja": "lotiti vprašanje",
    "spremeniti vir": "spremeniti vire",
    "analizirati program": "analizirati programe",
    "olajšati pogovor": "olajšati pogovore",
    "ugotoviti primer": "ugotoviti primere",
    "izdajati sklep": "izdajati sklepe",
    "zaslediti primer": "zaslediti primere",
    "prevzeti klic": "prevzeti klice",
    "vključiti vir": "vključiti vire",
    "vključiti uporabnika": "vključiti uporabnike",
    "spremeniti trg": "spremeniti trge",
    "nabrati Izkušnje": "nabrati izkušnjo",
    "iskati izhod": "iskati izhode",
    "nabirati izkušnje": "nabirati izkušnjo",
    "iskati prostor": "iskati prostore",
    "aktivirati proces": "aktivirati procese",
    "iskati pristop": "iskati pristope",
    "zaprositi oblast": "zaprositi oblasti",
    "zapisati sanje": "zapisati sanjo",
    "prejeti prepoved": "prejeti prepovedi",
    "služiti življenju": "služiti življenje",
    "uničiti sanje": "uničiti sanjo",
    "menjati sistem": "menjati sisteme",
    "kazati nivo": "kazati nivoje",
    "napasti sistem": "napasti sisteme",
    "iskati modrost": "iskati modrosti",
    "zapisati vtis": "zapisati vtise",
    "odpreti vpogled": "odpreti vpoglede",
    "aktivirati sposobnost": "aktivirati sposobnosti",
    "najti povzetek": "najti povzetke",
    "jesti obrok": "jesti obroke",
    "prejeti dokument": "prejeti dokumente",
    "najti žival": "najti živali",
    "iskati znak": "iskati znake",
    "kazati znak": "kazati znake",
    "iskati recept": "iskati recepte",
    "uničiti pridelek": "uničiti pridelke",
    "vpisati znak": "vpisati znake",
    "mešati karte": "mešati karto",
    "prehiteti kolesarja": "prehiteti kolesara",
    "mešati jajca": "mešati jajce",
    "narezati jajca": "narezati jajce",
    "iskati jajca": "iskati jajce",
    "prodati hotel": "prodati hotele",
    "krasiti model": "krasiti modele",
    "kupovati model": "kupovati modele",
    "prodati pridelek": "prodati pridelke",
    "nabrati zelišča": "nabrati zelišče",
    "ležati jajca": "ležati jajce",
    "iskati žival": "iskati živali",
    "kupovati jajca": "kupovati jajce",
    "najti planet": "najti planete",
    "nabirati jagode": "nabirati jagodo",
    "naročiti obrok": "naročiti obroke",
    "umakniti podpis": "umakniti podpise",
    "objaviti sklep": "objaviti sklepe",
    "prebrati napoved": "prebrati napovedi",
    "opraviti vpis": "opraviti vpise",
    "reševati primer": "reševati primere",
    "odigrati turnir": "odigrati turnirje",
    "objaviti pesem": "objaviti pesmi",
    "pisati program": "pisati programe",
    "igrati pesem": "igrati pesmi",
    "prebrati program": "prebrati programe",
    "opraviti sestanek": "opraviti sestanke",
    "iskati primer": "iskati primere",
    "prejeti klic": "prejeti klice",
    "narisati načrt": "narisati načrte",
    "iskati vir": "iskati vire",
    "aktivirati mehanizem": "aktivirati mehanizme",
    "posneti prizor": "posneti prizore",
    "nanesti odmerek": "nanesti odmerka",
    "zapreti odsek": "zapreti odseke",
    "ustaviti načrt": "ustaviti načrte",
    "vaditi pesem": "vaditi pesmi",
    "najti portal": "najti portale",
    "najti mehanizem": "najti mehanizme",
    "urediti pesem": "urediti pesmi",
    "ustaviti tok": "ustaviti tokove",
    "kupovati program": "kupovati programe",
    "vpisati predlog": "vpisati predloge",
    "nabrati sestavine": "nabrati sestavino",
    "posneti program": "posneti programe",
    "našteti napake": "našteti napako",
    "odpreti vir": "odpreti vire",
    "zbrati vprašanja": "zbrati vprašanje",
    "iskati pesem": "iskati pesmi",
    "iskati trg": "iskati trge",
    "načrtovati sestanek": "načrtovati sestanke",
    "izračunati prihranek": "izračunati prihranke",
    "kupovati sestavine": "kupovati sestavino",
    "obvestiti vodjo": "obvestiti vodje",
    "pozdraviti predstavnika": "pozdraviti predstavnike",
    "najti sponzorja": "najti sponzora",
    "nabrati kilogram": "nabrati kilograma",
    "vpisati zadetek": "vpisati zadetka",
    "prodati igralca": "prodati igralce",
    "kupovati igralca": "kupovati igralce",
    "iskati avtorja": "iskati avtora",
    "objaviti izid": "objaviti izide",
    "odpreti link": "odpreti linka",
    "prebrati sklep": "prebrati sklepa",
    "iskati arhitekta": "iskati arhitekte",
    "posneti koncert": "posneti koncerte",
}
