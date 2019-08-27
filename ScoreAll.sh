cd Evaluation_Datasets

grep senseval2 ../$1 | awk '{gsub("senseval2.","");printf("%s %s\n",$1,$2);}' > TEMP
echo -e '-------------------------------------------\nsenseval2'
java Scorer senseval2/senseval2.gold.key.txt TEMP 

grep senseval3 ../$1 | awk '{gsub("senseval3.","");printf("%s %s\n",$1,$2);}' > TEMP
echo -e '-------------------------------------------\nsenseval3'
java Scorer senseval3/senseval3.gold.key.txt TEMP 

grep semeval2013 ../$1 | awk '{gsub("semeval2013.","");printf("%s %s\n",$1,$2);}' > TEMP
echo -e '-------------------------------------------\nsemeval2013'
java Scorer semeval2013/semeval2013.gold.key.txt TEMP 

grep semeval2015 ../$1 | awk '{gsub("semeval2015.","");printf("%s %s\n",$1,$2);}' > TEMP
echo -e '-------------------------------------------\nsemeval2015'
java Scorer semeval2015/semeval2015.gold.key.txt TEMP 

grep -v semeval2007 ../$1 > TEMP
echo -e '-------------------------------------------\nALL_4'
java Scorer ALL_4.gold.key.txt TEMP

rm TEMP
cd ..
