cnt(){ powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? {\$_.CommandLine -match 'answer_eval' -and \$_.CommandLine -match '--name [A-Za-z0-9-]*-base'}).Count"; }
for j in E3-b-base E3-a-base E4-a-base E6-a-base; do
  while [ "$(cnt)" -ge 5 ]; do sleep 10; done
  if [ $j = E3-b-base ]; then
    ( bash $L/ans_jobs.sh $j; ls $RR | grep "_rag_E3-b-base-similarity" | head -1 > $L/E3BA.txt ) &
  else
    bash $L/ans_jobs.sh $j &
  fi
  sleep 30
done
wait
echo done > $L/rerun_done
