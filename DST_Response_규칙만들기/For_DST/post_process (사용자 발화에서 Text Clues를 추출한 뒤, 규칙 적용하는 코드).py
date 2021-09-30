import re
from tqdm import tqdm
# 전체 대화 관점에서 매칭 되는지
# 각 Turn 관점에서 매칭 되는지

total_turn = 0
keyword_in_slots = 0
keyword_not_slots = 0

Keywords = {
    'type' : ['trousers', 'blouse', 'sweater', 'joggers', 'dress', 'hat', 'coat', 'jeans', 'tshirt', 'jacket', 'shoes', 'vest', 'tank top', 'suit', 'shirt', 'hoodie',
            'Bed', 'CoffeeTable', 'CouchChair', 'Shelves', 'EndTable', 'Sofa', 'AreaRug', 'Lamp', 'Chair', 'Table'],
    'brand' : ['HairDo', 'Garden Retail', 'Yogi Fit', 'Brain Puzzles', 'Uptown Gallery', '212 Local', 'Art Den', 'Glam Nails', 'Art News Today', 'Nature Photographers', 'Downtown Consignment', 'StyleNow Feed', 'Home Store', 'Ocean Wears', 'Global Voyager', 'Uptown Studio', 'North Lodge', 'New Fashion', 'Pedals & Gears', 'Downtown Stylists', 'Cats Are Great', 'Modern Arts', 'River Chateau', 'Fancy Nails', 'The Vegan Baker', 'Coats & More',\
                'StyleNow Feed', 'Modern Arts', 'Uptown Studio', 'North Lodge', 'Home Store', 'River Chateau', 'Uptown Gallery', 'Art Den', '212 Local', 'Global Voyager', 'Downtown Consignment', 'Downtown Stylists'],
    'pattern' :  ['leafy design', 'stripes', 'plain', 'text', 'diamonds', 'vertical striples', 'multicolored', 'checkered', 'light spots', 'radiant', 'plain with stripes on side', 'checkered, plain', 'denim', 'light vertical stripes', 'horizontal stripes', 'floral', 'vertical stripes', 'design', 'velvet', 'spots', 'heavy vertical stripes', 'twin colors', 'knit', 'dotted', 'light stripes', 'leather', 'vertical design', 'holiday', 'heavy stripes', 'canvas', 'plaid', 'cargo', 'camouflage'],
    'color' : ['orange', 'dark blue', 'light blue', 'violet', 'purple', 'grey', 'beige', 'dark red', 'olive', 'maroon', 'white', 'pink', 'dark green', 'dirty grey', 'black', 'red', 'light pink', 'light red', 'dark brown', 'dark grey', 'brown', 'light green', 'dark yellow', 'light grey', 'dirty green', 'dark pink', 'dark violet', 'light orange', 'golden', 'blue', 'yellow', 'green',\
                'brown', 'black and white', 'grey', 'black', 'red', 'wooden', 'blue', 'green', 'white'],
    'sleeveLength' : ['short', 'long', 'full', 'sleeveless', 'half']    ,
    'price' : ['129.99', '4.99', '134.99', '9.99', '139.99', '14.99', '144.99', '19.99', 'expensive', '149.99', '24.99', '154.99', '29.99', '34.99', '164.99', '39.99', '169.99', '44.99', '174.99', '49.99', '179.99', '54.99', '184.99', '59.99', '189.99', '64.99', '69.99', '199.99', '74.99', '204.99', '79.99', '209.99', '84.99', '214.99', '89.99', 'cheap', '94.99', '224.99', '99.99', '229.99', '234.99', '109.99', '239.99', '114.99', '244.99', '124.99', 'affordable',\
                '$299', '$549', '$399', '$449', '$649', 'expensive', '$499', '$349', 'cheap', '$599', '$249', '$199', 'affordable'],
    'size' : ['L', 'M', 'XS', 'XXL', 'XL', 'S'],        
    'customerReview' : ['good'],
    'customerRating ' : ['good'],
    'materials' : ['wool', 'memory foam', 'leather', 'natural fibers', 'metal', 'wood'],
}

REPLACE_WORD_l1 = {
    'nice':'good',      # customerReview
    'favorable':'good', # customerReview
    'kinda':'good',

    'rug':'AreaRug',    # type
    'sweatpants':'joggers',
    'pants':'trousers',
    'gray':'grey',
    'bottoms' : 'trousers',

    't-shirt':'tshirt',
    'tshirts':'tshirt',    
    'blouses':'blouse',
    'dresses':'dress',
    'jackets':'jacket',
    'hats':'hat',
    'coats':'coat',
    'sweaters':'sweater',
    'shirts':'shirt',
    'suits':'suit',
    'hoodies':'hoodie',
    'vests':'vest',
    'sofas':'sofa',

    'priced':'expensive',
    'fancy' :'expensive',
    'pricier':'expensive',
    'pricey':'expensive',

}

REPLACE_WORD_l2 = {
    'extra large' : 'XL',   # 단순히 large는 XL, L 모두 의미
    'extra small' : 'XS',   # 단순히 small은 XS, S 모두 의미
    "n't expensive" : 'affordable',
    "not expensive" : 'affordable',
    'soft stripes' : 'light stripes',
    'soft dotted' : 'light spots',

}

REPLACE_WORD_l3 = {
    'extra extra large' : 'XXL',   # 단순히 large는 XL, L 모두 의미
    "n't too expensive" : 'affordable',
    "not too expensive" : 'affordable',
}
def cleanText(readData):
 
    #텍스트에 포함되어 있는 특수 문자 제거
 
    text = re.sub('[=+,#/\?:^.@*\"※~%ㆍ!』\\‘|\(\)\[\]\<\>`…》]', '', readData)
    text = text.replace('-', ' ') 
    return text


def get_ngram(tokens, n):
    grams = list()
    for num in range(len(tokens)):  
        if num+n > len(tokens): break
        grams.append((" ").join(tokens[num:num+n]))
    return grams.copy()

def ngram(text, n=5):
    grams = dict()

    for key, value in REPLACE_WORD_l3.items():  text = text.replace(key, value)
    for key, value in REPLACE_WORD_l2.items():  text = text.replace(key, value)
    text = text.split(" ")
    text = [REPLACE_WORD_l1[token] if token in REPLACE_WORD_l1 else token for token in text]

    for num in range(n):
        num += 1
        if not num in grams.keys():
            grams[num] = list()
        if num == 1:    grams[num] = text.copy()
        else:           grams[num] = get_ngram(text, num)
    return grams



def extractKeyword(readData):


    # rating, sizing에 대한 확인

    # Hm. How do the red and white jacket and the blue jacket to the right of it on the central rack compare on review and available size?
    # user_belief : {}
    # 위 경우처럼, available size를 묻지만, 정작 belief state는 잡히지 않은 경우도 있다

    # user utter : What would you recommend?
    # user_belief : {'pattern': 'design', 'size': 'S', 'price': 59.99, 'sleeveLength': 'full', 'type': 'coat'}
    # 이런걸 어떻게 잡아내냐
    # rcommend 라는 표현에 뭐가 있나?

    # user utter : OK, can you show me a nice plain-color jacket?
    # user_belief : {'type': 'coat', 'pattern': 'plain'}
    # jacket을 이야기했는데, 왜 coat를 잡냐

    # user uttr : Thanks I also want a long sleeve jacket in extra large
    # user_belief : {'type': 'coat', 'size': 'XL', 'sleeveLength': 'full'}
    # 아니 시발 long이 왜 full로 바껴

    # pattern이나 color는 대게 type 앞에 위치한다.
    words = dict()

    # slot values가 2개 이상 잡히는 경우가 여전히 있다.

    for slot_type in Keywords:
        for slot_value in Keywords[slot_type]:
            for idx, ngram in enumerate(readData.values()):
                for seq in ngram:
                    # stemmer는 너무 느릴 뿐만 아니라, 완벽하지도 않다.
                    # 가령 blouses -> blous로 바꾸는 문제가 있다.
                    if slot_value.lower() == seq.lower():
                        if not slot_type in words:
                            words[slot_type] = set()
                        words[slot_type].add(slot_value)

    return words.copy()

def slots_recheck(user_uttr=None, pred_slots=None, n=5):
    user_uttr = cleanText(user_uttr)
    user_utter_ngram = ngram(user_uttr, 5)
    extractSlotsKeywords = extractKeyword(user_utter_ngram)

    modified_slots = list()
    for [slot_type, slot_value] in pred_slots:
        if slot_type in extractSlotsKeywords:
            # 복수의 값
            if str(type(extractSlotsKeywords[slot_type])) == "<class 'list'>":
                if not slot_value in extractSlotsKeywords[slot_type]:
                    slot_value = extractSlotsKeywords[slot_type]
            # 단일 값
            else:
                if not slot_value == extractSlotsKeywords[slot_type]:
                    slot_value = extractSlotsKeywords[slot_type]
        modified_slots.append([slot_type,slot_value])

    return modified_slots.copy()