en_prompt = """Background: A concrete expression refers to an event with which you can have an immediate experience through your senses (smelling, tasting, touching, hearing, seeing) and by performing or witnessing the event. An example of a concrete expression is 'drive car'.
An abstract expression refers to an event or process you cannot experience directly through your senses or by performing the event. An example of an abstract expression is 'deliver justice'.

Task: Rate the concreteness of the following expression on a scale from 1.0 to 5.0 (using one decimal place), where:
1.0 = very abstract
5.0 = very concrete

Focus on only responding with floating-point number rating, and do not add any explanations. The values have to be within the 1.0 and 5.0 range.
"""

de_prompt = """Hintergrund: Ein konkreter Ausdruck bezieht sich auf ein Ereignis, das Sie mit Ihren Sinnen (Riechen, Schmecken, Tasten, Hören, Sehen) und durch Ausführen oder Miterleben des Ereignisses unmittelbar erfahren können. Ein Beispiel für einen konkreten Ausdruck ist „Auto fahren“.
Ein abstrakter Ausdruck bezieht sich auf ein Ereignis oder einen Vorgang, den Sie nicht direkt mit Ihren Sinnen oder durch Ausführen des Ereignisses erfahren können. Ein Beispiel für einen abstrakten Ausdruck ist „Recht sprechen“.

Aufgabe: Bewerten Sie die Konkretheit des folgenden Ausdrucks auf einer Skala von 1.0 bis 5.0 (mit einer Nachkommastelle), wobei:
1.0 = sehr abstrakt
5.0 = sehr konkret

Konzentrieren Sie sich darauf, nur mit Gleitkommazahlen zu antworten, und fügen Sie keine Erklärungen hinzu. Die Werte müssen im Bereich zwischen 1,0 und 5,0 liegen.
"""

sl_prompt = """Ozadje: Konkretni izraz se nanaša na dogodek, ki ga lahko neposredno doživite s svojimi čutili (vonj, okus, dotik, sluh, vid) in z izvedbo ali pričo dogodka. Primer konkretnega izraza je „voziti avto“.
Abstraktni izraz se nanaša na dogodek ali proces, ki ga ne morete neposredno doživeti s svojimi čutili ali z izvedbo dogodka. Primer abstraktnega izraza je „zagotoviti pravico“.

Naloga: Ocenite konkretnost naslednjega izraza na lestvici od 1.0 do 5.0 (z enim decimalnim mestom), kjer:
1.0 = zelo abstrakten
5.0 = zelo konkreten

Osredotočite se le na odgovore z oceno števila s plavajočo vejico in ne dodajajte nobenih pojasnil. Vrednosti morajo biti znotraj razpona 1,0 in 5,0.
"""


def en_prompt_zero_shot(target):
    return (
        en_prompt
        + f"""
Now for the following expression, provide a concreteness rating:
Expression: '{target}'
Rating:"""
    )


def en_prompt_few_shot(target):
    return (
        en_prompt
        + f"""
Here are examples of expressions with their concreteness ratings:
Expression: 'embody principle'
Rating: 1.8

Expression: 'know cat'
Rating: 3.6

Expression: 'carry handgun'
Rating: 4.4

Expression: 'exceed budget'
Rating: 3.6

Expression: 'illustrate difficulty'
Rating: 2.8

Expression: 'consume energy'
Rating: 3.7

Now for the following expression, provide a concreteness rating:
Expression: '{target}'
Rating:"""
    )


def de_prompt_zero_shot(target):
    return (
        de_prompt
        + f"""
Geben Sie nun für den folgenden Ausdruck eine Konkretheitsbewertung ab:
Ausdruck: '{target}'
Bewertung:"""
    )


def de_prompt_few_shot(target):
    return (
        de_prompt
        + f"""
Hier finden Sie Beispiele für Ausdrücke mit ihren Konkretheitsbewertungen:
Ausdruck: 'Diskussion zulassen'
Bewertung: 2.5

Ausdruck: 'Bedeutung vermitteln'
Bewertung: 1.9

Ausdruck: 'Album planen'
Bewertung: 3.3

Ausdruck: 'Buch tragen'
Bewertung: 4.8

Ausdruck: 'Charakter unterstützen'
Bewertung: 2.3

Ausdruck: 'Entlassung ankündigen'
Bewertung: 3.5

Geben Sie nun für den folgenden Ausdruck eine Konkretheitsbewertung ab:
Ausdruck: '{target}'
Bewertung:"""
    )


def sl_prompt_zero_shot(target):
    return (
        sl_prompt
        + f"""
Zdaj za naslednji izraz navedite oceno konkretnosti:
Izraz: '{target}'
Ocena:"""
    )


def sl_prompt_few_shot(target):
    return (
        sl_prompt
        + f"""
Tukaj so primeri izrazov z njihovimi ocenami konkretnosti:
Izraz: 'napovedati vreme'
Ocena: 2.5

Izraz: 'odreči pravici'
Ocena: 1.3

Izraz: 'nasloviti pismo'
Ocena: 3.3

Izraz: 'piti kavo'
Ocena: 5.0

Izraz: 'izplačati znesek'
Ocena: 3.3

Izraz: 'opraviti registracijo'
Ocena: 3.6

Zdaj za naslednji izraz navedite oceno konkretnosti:
Izraz: '{target}'
Ocena:"""
    )


PROMPTS = {
    "en": (en_prompt_zero_shot, en_prompt_few_shot),
    "de": (de_prompt_zero_shot, de_prompt_few_shot),
    "sl": (sl_prompt_zero_shot, sl_prompt_few_shot),
}
