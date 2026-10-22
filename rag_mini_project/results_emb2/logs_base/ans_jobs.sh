B0A=20261022_100517_rag_B0-base-similarity
a(){ n=$1; shift; python -m rag.answer_eval --workers 12 --name $n --baseline $BL "$@" > $L/ans_$n.log 2>&1; }
BL=$B0A
PC=base_pc_c500_o50; ST=base_struct_c500_o50; RC=base_rec_c500_o50
case $1 in
E3-b-base) a E3-b-base --collection $PC --note "부모 문서 반환(E3-b)";;
E3-a-base) a E3-a-base --collection $ST --note "구조 단위 청킹(E3-a)";;
E4-a-base) a E4-a-base --collection base_rec_title_c500_o50 --note "제목 접두어(E4-a)";;
E6-a-base) a E6-a-base --collection $RC --k 3 --note "k=3";;
E6-b-base) a E6-b-base --collection $RC --k 10 --note "k=10";;
B0-base-rep2) a B0-base-rep2 --collection $RC --note "기준 B0 반복 2";;
B0-base-rep3) a B0-base-rep3 --collection $RC --note "기준 B0 반복 3";;
E3-a-base-rep2) a E3-a-base-rep2 --collection $ST --note "E3-a 반복 2";;
E3-b-base-rep2) a E3-b-base-rep2 --collection $PC --note "E3-b 반복 2";;
E7-base-rep1) a E7-base-rep1 --mode agent --collection $RC --note "Agent 반복 1";;
E7-base-rep2) a E7-base-rep2 --mode agent --collection $RC --note "Agent 반복 2";;
E7a-base) a E7a-base --mode agent --agent-options retry=0 --collection $RC --note "Agent 재시도 끔";;
E7b-base) a E7b-base --mode agent --agent-options analyze=0,route=0 --collection $RC --note "Agent 분석·라우팅 끔";;
E8*)
  until [ -f $L/E3BA.txt ]; do sleep 10; done
  BL=$(cat $L/E3BA.txt)
  k=3; case $1 in E8b*) k=10;; esac
  a $1 --collection $PC --k $k --note "부모 문서 반환 k=$k"
  ;;
esac
