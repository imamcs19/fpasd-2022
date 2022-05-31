#dictionary of common contractions and slang
#needed for textblob analysis
SLANG_DICT = {"cant": "can not", "im": "i am", "arent": "are not", "couldnt": "could not", "hasnt": "has not",
              "havent": "have not", "didnt": "did not", "dont": "do not", "doesnt": "does not", "aint": "am not",
              "id": "i would", "ill": "i will", "hadnt": "had not", "isnt": "is not", "whyd": "why did",
              "wasnt": "was not", "wont": "will not", "werent":"were not", "whats":"what is", "wouldnt":"would not",
              "youve":"you have", "thats":"that is", "ive": "i have", "wtf": "what the fuck", "u": "you", "r":"are",
              "wth":"what the hell", "ya":"yeah", "k":"ok", "lol":"laugh out loud", "lmao": "laughing my ass off",
              "af": "a lot", "lit": "amazing", "woke": "aware", "shade": "disrespect", "fr": "for real" }

#limit location to united states
LOCATIONS = [-124.85, 24.39, -66.88, 49.38,]

#common words to track on twitter
COMMON_WORDS = ["a", "the", "i", "you", "u", "of", "it", "my", "at", "that", "and", "s", "with", "do",
                "have", "just", "this", "be", "in", "is", "to", "on", "not", "was", "but", "what", "like",
                "good", "if", "as", "how", "can", "they", "am", "by", "go", "has", "rt", "know", "there",
                "more", "too", "he", "she", "did", "when", "really", "off", "would", "people", "america",
                "want", "why", "much", "only", "very", "than", "again", "news", "fake", "any", "where", "take"
                "said", "were", "world", "n", "say", "in", "today", "day", "said", "more", "give", "did",
                "me", "does", "its", "too", "to", "since", "only", "russia", "trump" ]

#HUGE list of stopwords and other random words that are common in tweets
STOPWORDS = ["a","able","about","above","abroad","absolutely","abt","according","accordingly","account","acne","across",
    "act","action","actually","add","adj","advocate","after","afterwards","ag","again","against","againwe","age","ago" "aint",
    "agree","ahead","all","alleged","allow","allowed","allows","almost","alone","along","alongside","alot","already","also",
    "although","always","am","amazing","america","american","americans","amid","amidst","among","amongst","amp","an","and",
    "another","any","anyhow","anymore","anything","apart","apparently","appear","appreciate","appropriate","are","arent","around",
    "art","arts","as","aside","ask","asked","asking","asks","ass","associate","associated","at","attitude","author","available",
    "aw","away","awe","awful","awfully","b","back","backward","backwards","bad","bag","be","became","because","become","becomes",
    "becoming","bed","been","before","beforehand","began","begin","beginning","behind","being","believe","below","beside","besides",
    "best","better","between","beyond","bf","big","biggest","billion","bio","bitch","bjp","black","blah","bless","blow","boom","both",
    "bottom","bought","boy","boys","brady","break","breakfast","breaking","brexit","brief","broke","btch","but","buy","by","c","call",
    "called","calling","calls","came","camera","camserotica","can","can't","cannot","cant","caption","care","carry","carrying","cause",
    "certain","certainly","change","changes","cheating","child","children","chill","chips","city","clap","class","clean","clear","clearly",
    "click","close","closes","cmon","co","com","come","comes","coming","companies","concern","concerning","consequently","consider",
    "considering","contain","containing","contains","continue","cook","cool","corresponding","could","couldnt","count","country","course",
    "cover","cr","cross","cry","cs","current","currently","cut","cuts","d","dare","darent","dark","day","days","ddrive","dear","def","define",
    "definitely","deletedlost","describe","described","describing","deserves","despite","developed","dick","did","didn","didnt","diff","different",
    "digital","direct","directly","dirty","disagree","disproportionately","do","does","doesn't","doesnt","doing","doj","don","done","dont","doo",
    "dowk","down","downwards","drive","drives","drop","dude","during","e","each","earn","easy","edu","eight","eighty","either","else","elsewhere",
    "em","end","ending","enough","entirely","especially","est","et","etc","even","event","ever","every","everyone","everything","ex","exactly",
    "example","except","exciting","f","face","fact","fake","fan","fans","far","farther","fast","feed","feel","feeling","felt","female","few",
    "find","first","five","follow","followers","following","for","forever","forget","form","former","formsubmission","forward","found","four",
    "free","from","front","fronts","fuck","fucking","full","fun","further","g","gain","garbage","gave","general","get","gets","getting","gf",
    "gift","girl","girls","give","glad","glass","go","god","goes","going","gone","good","goodmorning","goodnight","google","got","grain","grandma", "great",
    "gt","guess","guy","guys","h","ha","had","hadnt","hai","half","happen","happened","happens","happy","hard","has","hasnt","hate","hates","hating",
    "hav","have","havent","having","he","heading","hear","hell","hello","help","her","here","heres","hers","herself","hes","hey","hi","high","him",
    "himself","his","hms","home","hope","hopefully","hour","hours","how","https","human","husband","i","i'm","icymi","id","idiot","idiots","ie","if",
    "ik","ill","im","important","in","including","inferior","inside","interest","interesting","into","is","isnt","it","itd","itll","its","itself","ive",
    "j","jan","join","jr","jumla","just","k","keep","kept","kids","kill","killed","kind","kinda","know","known","knows","l","ladies","lady","lakh","last",
    "later","latest","laugh","learn","least","leave","left","less","let","lets","like","liked","likes","limitless","link","lips","list","little","live",
    "living","ll","local","lol","longway","look","looked","looking","looks","lost","lot","loud","love","loved","low","lunch","m","ma","mad","made",
    "main","make","makes","male","man","many","match","matching","maths","may","maybe","me","mean","meet","message","messages","met","might","million",
    "min","mine","mins","minute","minutes","miss","missing","moment","month","months","more","morning","most","mr","mrs","much","must","my","myself",
    "n","nd","near","necessary","need","needs","neither","network","neuro","never","new","news","next","night","nine","no","none","nor","normal","not",
    "notabot","nothing","now","nqame","number","nyum","o","obvious","of","off","office","often","oh","ok","okay","old","on","once","one","ones","only",
    "open","option","or","other","our","ours","ourselves","out","outside","over","own","p","page","paid","paint","pants","part","particular","party","pay",
    "people","per","piano","pic","pill","place","plan","play","playing","please","pls","plus","plz","pm","pod","point","pool","poor","possible","pray",
    "prayer","prime","private","probably","problem","provide","public","pump","put","putting","q","question","quiet","quite","r","ran","rap","rapping",
    "rather","ray","rays","rd","re","read","real","really","reason","receiv","receive","recent","remember","rent","return","retweet","right","rise",
    "round","rs","rt","rtd","s","said","same","save","saw","say","saying","says","screaming","second","see","seem","seems","seen","self","sent","serious",
    "setting","seven","several","sex","shah","shall","share","she","shed","sheets","shirt","shit","shocking","shoudlnt","should","show","shows", "since","singing",
    "six","smh","snapchat","so","solar","some","somebody","someone","something","sometimes","soon","spaghetti","speak","specific","spoke","spread","st",
    "stan","start","static","stay","step","stepping","stfu","still","stop","store","stores","strong","stupid","sub","submission","such","sup","superior",
    "support","sure","t","take","taken","takes","taking","talk","tea","team","tell","text","tf","th","than","thank","thanks","thanx","thar","that","thats",
    "the","thedeepgain","thee","their","theirs","them","themselves","then","there","theres","these","they","theyre","thing","things","think","third","this",
    "tho","those","thought","three","through","thru","tht","thts","thx","tide","tidepod","time","to","today","told","tom","tonight","too","took","tool","tools",
    "top","tought","toward","transformation","tried","true","try","trying","ts","turned","tweet","tweeted","twice","twitter","two","u","uk","un","under",
    "understand","unless","until","up","ur","us","use","used","uses","using","usually","v","vague","ve","very","via","video","vs","w","waah","wah","wait",
    "waiting","wanna","want","wants","was","wasnt","watch","wave","way","we","weak","wear","week","well","went","were","weve","what","whatever","whats",
    "when","whenever","where","wheres","which","while","white","who","whole","whom","whos","why","wife","will","willing","with","within","without","won",
    "wonder","wont","word","work", "working", "world","would","wouldnt","write","wrong","x","y","yeah","year","years","yep","yes","yet","you","youll","young",
    "your", "youre","yours","yourself","yourselves","youve","z","zero","zoom", "bts", "btstwt", "suddenly", "anyone", "friend", "friendly", "rest", "middle",
    "pts", "doesn", "wouldn", "btg", "ehh", "ca", "played", "rest", "feast", "ko", "cheat", "cheated", "shaku", "special", "shoes", "chess", "red", "act", "acting",
    "kick", "swamy", "bot", "given", "group", "daughter", "ago", "gummy", "sos", "tired", "therefore", "groups", "pdp",]

STOPWORDS_SET = set(STOPWORDS)