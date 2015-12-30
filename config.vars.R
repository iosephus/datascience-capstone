
data.dir <- "C:\\Users\\JoseM\\Projects\\Capstone\\Data"
#data.dir <- "~/Projects/Capstone/Data"
corpus.dir <- file.path(data.dir, "corpora", "en_US")
corpus.file.pattern = "*.training.txt"
sentences.file <- file.path(data.dir, "sentences.training.txt")

text.encoding = "UTF-8"
text.language = substring(basename(corpus.dir), 1, 2)
min.word.len = 1
max.word.len = 24
line.factor = 0.03
token.sentence.start = '<s>'
token.sentence.end = '</s>'
token.unknown = '<unk>'
token.num = '<num>'
num.cores = 12

