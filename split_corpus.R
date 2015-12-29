
require(tools)
source("config.vars.R")
source("common.R")

input.corpus.dir <- "C:\\Users\\JoseM\\Projects\\Capstone\\Corpus\\Coursera-SwiftKey\\final\\en_US"

file.list = list.files(input.corpus.dir, "*.txt", full.names=TRUE)

text.lines <- lapply(file.list, FUN=read.corpus.file)
names(text.lines) <- as.vector(sapply(file.list, function (path) file_path_sans_ext(basename(path))))

training.corpus.size = 0.8

set.seed(34093610)
selector.training <- lapply(text.lines, function (l) runif(length(l), 0.0, 1.0) < training.corpus.size)

training.lines = mapply(FUN = function(l, selector) l[selector], text.lines, selector.training)
testing.lines = mapply(FUN = function(l, selector) l[!selector], text.lines, selector.training)


if (!file.exists(corpus.dir)) {
    dir.create(corpus.dir, recursive=TRUE)
}

saveRDS(selector.training, file.path(corpus.dir, "training.lines.selector.rds"))

write.lines.to.file <- function(dir, suffix, extension, l_name, l_content) {
    full.file.path <- file.path(dir, paste(l_name, suffix, extension, sep="."))
    conn = file(full.file.path, encoding = text.encoding)
    writeLines(l_content, conn)
    close(conn)
}

mapply(function (l_name, l_content) write.lines.to.file(corpus.dir, "training", "txt", l_name, l_content), names(training.lines), training.lines)
mapply(function (l_name, l_content) write.lines.to.file(corpus.dir, "testing", "txt", l_name, l_content), names(testing.lines), testing.lines)
