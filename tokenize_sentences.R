
source("config.vars.R")
source("common.R")

require(quanteda)
require(snow)

message("Retrieving file list")
file.list = list.files(corpus.dir, corpus.file.pattern, full.names=TRUE)

message("Reading files")
text.lines = Reduce(c, as.vector(lapply(file.list, FUN=read.corpus.file)))

message("Tokenizning sentences")
sentences <- tokenize(text.lines, what="sentence", simplify=TRUE)

message("Saving sentences to file")
conn = file(sentences.file, encoding = text.encoding)
writeLines(sentences, conn)
close(conn)
