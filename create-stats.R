
source("config.vars.R")
source("explore.R")


data.raw <- load.corpus.content(corpus.dir, max.lines=line.limit, encoding=text.encoding)
file.composition <- get.pattern.stats(data.raw)
saveRDS(file.composition, file.path(data.dir, "file.composition.rds"))

content <- fix.unicode(do.call(paste, as.list(data.raw$content)))
corpus <- create.corpus(content)

dtm.1 <- DocumentTermMatrix(corpus, control=list(wordLengths=c(min.word.len, max.word.len)))
ngram.1.freq <- get.ngram.freq(dtm.1)
saveRDS(ngram.1.freq, file.path(data.dir, "ngram.1.freq.rds"))

dtm.2 <- DocumentTermMatrix(corpus, control = list(tokenize = NGramTokenizerFuncBuilder(2)))
ngram.2.freq <- get.ngram.freq(dtm.2)

saveRDS(ngram.1.freq, file.path(data.dir, "ngram.2.freq.rds"))

dtm.3 <- DocumentTermMatrix(corpus, control = list(tokenize = NGramTokenizerFuncBuilder(3)))
ngram.3.freq <- get.ngram.freq(dtm.3)

saveRDS(ngram.3.freq, file.path(data.dir, "ngram.3.freq.rds"))
