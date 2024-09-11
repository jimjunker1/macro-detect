i_am("code/clean-classifications.R")
# this script takes annotated images from the
# data/images and data/labels folders and determines
# the classification of each image and corresponding
# YOLO .txt files to allow for classification of multiple taxa

onedrivePath = sprintf("C:/Users/%s/OneDrive - UNT System/Food Web Grant Brainstorming/COS Seed Grant/COS-CENG Seed Grant 2024/",
        Sys.info()[['user']])

metadata = read.table(file = paste0(onedrivePath,"data/raw images/metadata.txt"), sep = ",", header = TRUE, strip.white = TRUE)

# recode the annotated .txt files to set the classInt based
annotated_files = list.files(path = paste0(onedrivePath,"data/annotated images/"), pattern = ".txt", recursive = TRUE, full.names = TRUE)
folder_paths = unique(gsub(paste0(onedrivePath,"data/annotated images/(.*_YOLO)/.*"),"\\1",annotated_files))
folder_paths = setNames(folder_paths, nm = folder_paths)
folder_map = purrr::map(folder_paths, \(x) annotated_files[grepl(x, annotated_files )])
taxaNames = purrr::map(names(folder_map), \(x) ifelse(lengths(strsplit(x, "_")) >4, paste(sapply(strsplit(x,"_"),"[", 2:3),collapse = " "), gsub(".*_(.*)_set0_YOLO","\\1",x)))
classInt = purrr::map_dbl(taxaNames, \(x) ifelse(lengths(strsplit(x," ")) > 1, metadata[grepl(x, metadata$taxaName),"classInt"], metadata[grepl(x, metadata$classID),"classInt"]))
file_rewrite = function(a, y = NULL){
  fileEmpty = tryCatch(ifelse(length(readLines(file(a))) == 0, TRUE, FALSE),
                       error = function(e) TRUE)
  if(fileEmpty){
    file.remove(a)
    return(NULL)
    } else{
  fileIn = read.table(a, sep = " ", header= FALSE)
  fileIn[1,1] <- y
  write.table(fileIn, a, append = FALSE, sep = " ", row.names = FALSE, col.names = FALSE)
    }
}
# walk across and rename the classInt
debugonce(file_rewrite)
purrr::walk2(folder_map, classInt, \(x,y) purrr::walk(x, \(a) file_rewrite(a, y)))
