ev(){ n=$1; c=$2; m=$3; note=$4; python -m rag.evaluate --name $n --collection $c --methods $m --baseline $R --note "$note" > $L/ret_$n.log 2>&1; }
case $1 in
E5-a-base) ev E5-a-base base_rec_c500_o50 bm25 "BM25 검색";;
E5-b-base) ev E5-b-base base_rec_c500_o50 hybrid "하이브리드 검색";;
E6-c-base) ev E6-c-base base_rec_c500_o50 mmr "MMR 검색";;
E1-a-base) ev E1-a-base base_rec_c300_o50 similarity "청크 크기 300";;
E1-b-base) ev E1-b-base base_rec_c800_o50 similarity "청크 크기 800";;
E2-a-base) ev E2-a-base base_rec_c500_o0 similarity "겹침 0";;
E2-b-base) ev E2-b-base base_rec_c500_o100 similarity "겹침 100";;
E4-a-base) ev E4-a-base base_rec_title_c500_o50 similarity "제목 접두어";;
E3-a-base) ev E3-a-base base_struct_c500_o50 similarity "구조 단위 청킹"
  A=$(ls -d $RR/*retrieval_E3-a-base-similarity | head -1 | xargs basename)
  python -m rag.evaluate --name E3-b-base --collection base_pc_c500_o50 --methods similarity --baseline $A --note "구조 단위를 parent로, child 500/50 검색 후 parent 반환(E3-a 대비)" > $L/ret_E3-b-base.log 2>&1;;
esac
