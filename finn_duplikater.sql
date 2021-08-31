;with dup as
(
	select *, row_number() over(partition by fk_Ansatt,Fk_dato,fk_stilling,fk_klient,fk_organisasjon,fk_fravaerskode,Fravaersprosent
									,over30dager,mistetdagsverk order by fk_Ansatt,Fk_dato,fk_stilling,fk_klient,fk_organisasjon,fk_fravaerskode, Fravaersprosent
									,over30dager,mistetdagsverk) rad
	FROM sykefravaer2
)
select count(*) 
from dup where rad>1