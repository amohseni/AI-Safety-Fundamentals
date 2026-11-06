# Builds the shared dataset of true and false statements used by pages 6.2, 6.3 and 7.2.
#
# The statement forms follow Marks and Tegmark (2023), "The Geometry of Truth" (cities,
# negated cities, Spanish-English translations), but every statement here is generated
# by this script: the city facts come from GeoNames (CC BY 4.0), through the file
# geonames.csv that Marks and Tegmark distribute in their repository
# (github.com/saprmarks/geometry-of-truth, commit 5d1c630); the translation and element
# lists are written below. No statement is copied from their datasets, which carry no license.
#
# Usage: python3 tools/truth/make_statements.py <geonames.csv> <out.json>
import csv, json, random, sys

SEED = 7
rng = random.Random(SEED)

# GeoNames country names, mapped to the plain names a reader expects. Countries not listed are dropped.
COUNTRY = {
    'China': 'China', 'India': 'India', 'Brazil': 'Brazil', 'United States': 'the United States', 'Japan': 'Japan',
    'Mexico': 'Mexico', 'Russian Federation': 'Russia', 'Indonesia': 'Indonesia', 'Turkey': 'Turkey',
    'Korea, Republic of': 'South Korea', 'Nigeria': 'Nigeria', 'Iran, Islamic Rep. of': 'Iran', 'Pakistan': 'Pakistan',
    'Colombia': 'Colombia', 'United Kingdom': 'the United Kingdom', 'Philippines': 'the Philippines', 'Egypt': 'Egypt',
    'South Africa': 'South Africa', 'Malaysia': 'Malaysia', 'Canada': 'Canada', 'Venezuela, Bolivarian Rep. of': 'Venezuela',
    'Ukraine': 'Ukraine', 'Iraq': 'Iraq', 'Morocco': 'Morocco', 'Spain': 'Spain', 'Saudi Arabia': 'Saudi Arabia',
    'Poland': 'Poland', 'Italy': 'Italy', 'Germany': 'Germany', 'Peru': 'Peru', 'Viet Nam': 'Vietnam',
    'Kazakhstan': 'Kazakhstan', 'Bangladesh': 'Bangladesh', 'Argentina': 'Argentina', 'Australia': 'Australia',
    'Ecuador': 'Ecuador', 'Ethiopia': 'Ethiopia', 'Kenya': 'Kenya', 'Romania': 'Romania', 'France': 'France',
    'Belarus': 'Belarus', 'Algeria': 'Algeria', 'United Arab Emirates': 'the United Arab Emirates', 'Cameroon': 'Cameroon',
    'Uzbekistan': 'Uzbekistan', 'New Zealand': 'New Zealand', 'Afghanistan': 'Afghanistan', 'Cuba': 'Cuba',
    'Senegal': 'Senegal', 'Ghana': 'Ghana', 'Netherlands': 'the Netherlands', 'Sweden': 'Sweden', 'Chile': 'Chile',
    'Bulgaria': 'Bulgaria', 'Jordan': 'Jordan', 'Israel': 'Israel', 'Belgium': 'Belgium', 'Czech Republic': 'the Czech Republic',
    'Greece': 'Greece', 'Thailand': 'Thailand', 'Austria': 'Austria', 'Portugal': 'Portugal', 'Ireland': 'Ireland',
    'Hungary': 'Hungary', 'Finland': 'Finland', 'Norway': 'Norway', 'Denmark': 'Denmark', 'Switzerland': 'Switzerland',
    'Serbia': 'Serbia', 'Croatia': 'Croatia', 'Uruguay': 'Uruguay', 'Lebanon': 'Lebanon', 'Tunisia': 'Tunisia',
    'Myanmar': 'Myanmar', 'Nepal': 'Nepal', 'Sri Lanka': 'Sri Lanka', 'Singapore': 'Singapore', 'Mongolia': 'Mongolia',
    'Georgia': 'Georgia', 'Armenia': 'Armenia', 'Latvia': 'Latvia', 'Estonia': 'Estonia', 'Lithuania': 'Lithuania',
    'Bolivia': 'Bolivia', 'Paraguay': 'Paraguay', 'Uganda': 'Uganda', 'Zimbabwe': 'Zimbabwe', 'Zambia': 'Zambia',
    'Angola': 'Angola', 'Syrian Arab Republic': 'Syria', 'Qatar': 'Qatar', 'Kuwait': 'Kuwait',
}
PER_COUNTRY = 4
# Districts of larger cities, names shared by well-known cities in other countries, and
# places a reader is unlikely to know.
EXCLUDE = {'Brooklyn', 'Iztapalapa', 'Soweto', 'Caloocan City', 'Quezon City', 'Petaling Jaya', 'Al Ahmadi',
           'Valencia', 'Santiago', 'Cordoba', 'Trujillo', 'Birmingham', 'Perth', 'Leon de los Aldama', 'Puente Alto',
           'Pikine', 'Callao', 'Shubra al Khaymah', 'Sale', 'Boumerdes', 'Tebessa', 'Kakamega', 'Klang', 'Bekasi',
           'Al Ain City', 'Cuenca', 'Salvador', 'Giza', 'Rangpur', 'Namangan', 'Zarqa', 'Irbid', 'Latakia', 'Erbil'}
N_CITIES = 220

# Spanish word, English meaning. Common words with one standard translation.
SPANISH = [
    ('uno', 'one'), ('dos', 'two'), ('tres', 'three'), ('cuatro', 'four'), ('cinco', 'five'), ('diez', 'ten'),
    ('perro', 'dog'), ('gato', 'cat'), ('caballo', 'horse'), ('pájaro', 'bird'), ('pez', 'fish'), ('vaca', 'cow'),
    ('casa', 'house'), ('agua', 'water'), ('fuego', 'fire'), ('tierra', 'earth'), ('cielo', 'sky'), ('sol', 'sun'),
    ('luna', 'moon'), ('estrella', 'star'), ('árbol', 'tree'), ('flor', 'flower'), ('mar', 'sea'), ('río', 'river'),
    ('montaña', 'mountain'), ('libro', 'book'), ('mesa', 'table'), ('silla', 'chair'), ('puerta', 'door'),
    ('ventana', 'window'), ('ciudad', 'city'), ('calle', 'street'), ('escuela', 'school'), ('iglesia', 'church'),
    ('pan', 'bread'), ('leche', 'milk'), ('queso', 'cheese'), ('manzana', 'apple'), ('rojo', 'red'), ('verde', 'green'),
    ('azul', 'blue'), ('blanco', 'white'), ('negro', 'black'), ('grande', 'big'), ('pequeño', 'small'),
    ('caliente', 'hot'), ('frío', 'cold'), ('nuevo', 'new'), ('viejo', 'old'), ('feliz', 'happy'), ('triste', 'sad'),
    ('hombre', 'man'), ('mujer', 'woman'), ('niño', 'boy'), ('madre', 'mother'), ('padre', 'father'),
    ('hermano', 'brother'), ('amigo', 'friend'), ('día', 'day'), ('noche', 'night'), ('año', 'year'),
    ('semana', 'week'), ('hablar', 'to speak'), ('comer', 'to eat'), ('beber', 'to drink'), ('dormir', 'to sleep'),
    ('correr', 'to run'), ('escribir', 'to write'), ('leer', 'to read'), ('cantar', 'to sing'), ('vivir', 'to live'),
    ('cabeza', 'head'), ('mano', 'hand'), ('corazón', 'heart'), ('ojo', 'eye'),
]

# Element name, chemical symbol.
ELEMENTS = [
    ('hydrogen', 'H'), ('helium', 'He'), ('lithium', 'Li'), ('carbon', 'C'), ('nitrogen', 'N'), ('oxygen', 'O'),
    ('fluorine', 'F'), ('neon', 'Ne'), ('sodium', 'Na'), ('magnesium', 'Mg'), ('aluminium', 'Al'), ('silicon', 'Si'),
    ('phosphorus', 'P'), ('sulfur', 'S'), ('chlorine', 'Cl'), ('argon', 'Ar'), ('potassium', 'K'), ('calcium', 'Ca'),
    ('titanium', 'Ti'), ('chromium', 'Cr'), ('manganese', 'Mn'), ('iron', 'Fe'), ('cobalt', 'Co'), ('nickel', 'Ni'),
    ('copper', 'Cu'), ('zinc', 'Zn'), ('arsenic', 'As'), ('bromine', 'Br'), ('krypton', 'Kr'), ('silver', 'Ag'),
    ('tin', 'Sn'), ('iodine', 'I'), ('xenon', 'Xe'), ('platinum', 'Pt'), ('gold', 'Au'), ('mercury', 'Hg'),
    ('lead', 'Pb'), ('uranium', 'U'), ('radon', 'Rn'), ('tungsten', 'W'), ('barium', 'Ba'), ('boron', 'B'),
    ('beryllium', 'Be'), ('plutonium', 'Pu'), ('radium', 'Ra'), ('cesium', 'Cs'), ('selenium', 'Se'), ('gallium', 'Ga'),
]


def other(pool, avoid):
    while True:
        x = rng.choice(pool)
        if x != avoid:
            return x


def cities(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    rows = [r for r in rows if r['Feature Code'] in ('PPLC', 'PPLA', 'PPL', 'PPLA2') and r['Country name EN'] in COUNTRY]
    rows.sort(key=lambda r: -int(r['Population'] or 0))
    seen, count, out = set(), {}, []
    for r in rows:
        name = r['ASCII Name'].strip()
        c = COUNTRY[r['Country name EN']]
        if name in seen or name in EXCLUDE or not name.isascii() or any(ch.isdigit() for ch in name):
            continue
        if count.get(c, 0) >= PER_COUNTRY:
            continue
        seen.add(name)
        count[c] = count.get(c, 0) + 1
        out.append((name, c, int(r['Population'])))
        if len(out) >= N_CITIES:
            break
    return out


def main(geo, outpath):
    S = []
    def add(text, label, topic, form, item, **kw):
        S.append(dict(id=len(S), text=text, label=label, topic=topic, form=form, item=item, **kw))

    cs = cities(geo)
    countries = sorted({c for _, c, _ in cs})
    for k, (name, c, pop) in enumerate(cs):
        wrong = other(countries, c)
        for y, lab in ((c, 1), (wrong, 0)):
            add(f'The city of {name} is in {y}.', lab, 'cities', 'plain', k, subj=name, obj=y)
            add(f'The city of {name} is not in {y}.', 1 - lab, 'cities', 'neg', k, subj=name, obj=y)
            add(f'{name} is a city in {y}.', lab, 'cities', 'para', k, subj=name, obj=y)
    meanings = [e for _, e in SPANISH]
    for k, (w, e) in enumerate(SPANISH):
        wrong = other(meanings, e)
        for y, lab in ((e, 1), (wrong, 0)):
            add(f"The Spanish word '{w}' means '{y}'.", lab, 'spanish', 'plain', k, subj=w, obj=y)
            add(f"The Spanish word '{w}' does not mean '{y}'.", 1 - lab, 'spanish', 'neg', k, subj=w, obj=y)
            add(f"The English word '{y}' is a translation of the Spanish word '{w}'.", lab, 'spanish', 'para', k, subj=w, obj=y)
    syms = [s for _, s in ELEMENTS]
    for k, (el, s) in enumerate(ELEMENTS):
        wrong = other(syms, s)
        for y, lab in ((s, 1), (wrong, 0)):
            add(f'The chemical symbol for {el} is {y}.', lab, 'elements', 'plain', k, subj=el, obj=y)
            add(f'The chemical symbol for {el} is not {y}.', 1 - lab, 'elements', 'neg', k, subj=el, obj=y)
            add(f'{y} is the chemical symbol for {el}.', lab, 'elements', 'para', k, subj=el, obj=y)
    # Alice's opinion, for contrast-consistent search: assigned at random, independent of truth.
    for s in S:
        s['alice'] = rng.randint(0, 1)
    json.dump({'seed': SEED, 'statements': S}, open(outpath, 'w'), ensure_ascii=False, indent=0)
    from collections import Counter
    print(len(S), Counter((s['topic'], s['form']) for s in S))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
