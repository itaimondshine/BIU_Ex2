from GASolver import GeneticSolver, SolverType
from pathlib import Path
import sys

#ciphertext = "wvjd fd kfyum v.p.q. kjyxlj, yq dyenuylcqe, c wyq pniv qeunio wcev ijueycd zyieq cd evj mcqeucknecfd fz evj cdvykceydeq fz qfnev ypjuciy, ydm cd evj xjflfxciyl ujlyecfdq fz evj sujqjde ef evj syqe cdvykceydeq fz evye ifdecdjde. evjqj zyieq qjjpjm ef pj ef evufw qfpj lcxve fd evj fucxcd fz qsjicjq evye paqejua fz paqejucjq, yq ce vyq kjjd iylljm ka fdj fz fnu xujyejqe svclfqfsvjuq evye pyda ydm xuyrj fkhjiecfdq pya kj ymrydijm yxycdqe evj evjfua fz mjqijde wcev pfmczciyecfd evufnxv dyenuyl qjljiecfd, c mf dfe mjda. c vyrj jdmjyrfnujm ef xcrj ef evjp evjcu znll zfuij. dfevcdx ye zcuqe iyd yssjyu pfuj mczzcinle ef kjlcjrj evyd evye evj pfuj ifpsljb fuxydq ydm cdqecdieq qvfnlm vyrj kjjd sjuzjiejm dfe ka pjydq qnsjucfu ef, evfnxv ydylfxfnq wcev, vnpyd ujyqfd, kne ka evj yiinpnlyecfd fz cddnpjuyklj qlcxve ryucyecfdq, jyiv xffm zfu evj cdmcrcmnyl sfqqjqqfu. djrjuevjljqq, evcq mczzcinlea, evfnxv yssjyucdx ef fnu cpyxcdyecfd cdqnsjuykla xujye, iyddfe kj ifdqcmjujm ujyl cz wj ympce evj zfllfwcdx sufsfqcecfdq, dypjla,  evye xuymyecfdq cd evj sjuzjiecfd fz yda fuxyd fu cdqecdie, wvciv wj pya ifdqcmju, jcevju mf dfw jbcqe fu ifnlm vyrj jbcqejm, jyiv xffm fz ceq ocdm,  evye yll fuxydq ydm cdqecdieq yuj, cd jrju qf qlcxve y mjxujj, ryucyklj,  ydm, lyqela, evye evjuj cq y qeunxxlj zfu jbcqejdij ljymcdx ef evj sujqjuryecfd fz jyiv sufzceyklj mjrcyecfd fz qeunienuj fu cdqecdie. evj eunev fz evjqj sufsfqcecfdq iyddfe, c evcdo, kj mcqsnejm"
ciphertext = Path(sys.argv[1]).read_text()
solver = GeneticSolver(ciphertext)
solver.verbose = True

plaintext = solver.solve(SolverType.REGULAR)
print(plaintext)
