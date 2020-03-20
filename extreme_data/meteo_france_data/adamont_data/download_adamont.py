import subprocess

requests = """https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/CNRM-CM5/ALADIN53/historical/day/snowswe/SNOWSWE_PRO_CNRM-ALADIN53_CNRM-CERFACS-CNRM-CM5_HISTO_alp_1950080106_2005073106_6h.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/IPSL-CM5A/WRF331F/historical/day/snowswe/SNOWSWE_PRO_IPSL-INERIS-WRF331F_IPSL-IPSL-CM5A-MR_HISTO_alp_1951080106_2005073106_6h.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/CNRM-CM5/RCA4/historical/day/snowswe/SNOWSWE_PRO_SMHI-RCA4_CNRM-CERFACS-CNRM-CM5_HISTO_alp_1970080106_2005073106_6h.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/EC-EARTH/RCA4/historical/day/snowswe/SNOWSWE_PRO_SMHI-RCA4_ICHEC-EC-EARTH_HISTO_alp_1970080106_2005073106_6h.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MOHC-HadGEM2/RCA4/historical/day/snowswe/SNOWSWE_PRO_SMHI-RCA4_MOHC-HadGEM2-ES_HISTO_alp_1981080106_2005073106_6h.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/IPSL-CM5A/RCA4/historical/day/snowswe/SNOWSWE_PRO_SMHI-RCA4_IPSL-IPSL-CM5A-MR_HISTO_alp_1970080106_2005073106_6h.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MPI-ESM-LR/RCA4/historical/day/snowswe/SNOWSWE_PRO_SMHI-RCA4_MPI-M-MPI-ESM-LR_HISTO_alp_1970080106_2005073106_6h.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MOHC-HadGEM2/RACMO22E/historical/day/snowswe/SNOWSWE_PRO_KNMI-RACMO22E_MOHC-HadGEM2-ES_HISTO_alp_1981080106_2005073106_6h.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/CNRM-CM5/CCLM4-8-17/historical/day/snowswe/SNOWSWE_PRO_CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5_HISTO_alp_1950080106_2005073106_6h.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/EC-EARTH/CCLM4-8-17/historical/day/snowswe/SNOWSWE_PRO_CLMcom-CCLM4-8-17_ICHEC-EC-EARTH_HISTO_alp_1950080106_2005073106_6h.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MOHC-HadGEM2/CCLM4-8-17/historical/day/snowswe/SNOWSWE_PRO_CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES_HISTO_alp_1981080106_2005073106_6h.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MPI-ESM-LR/CCLM4-8-17/historical/day/snowswe/SNOWSWE_PRO_CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR_HISTO_alp_1950080106_2005073106_6h.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MPI-ESM-LR/REMO019/historical/day/snowswe/SNOWSWE_PRO_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_HISTO_alp_1950080106_2005073106_6h.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/NorESM1/HIRHAM5/historical/day/snowswe/SNOWSWE_PRO_DMI-HIRHAM5_NCC-NorESM1-M_HISTO_alp_1951080106_2005073106_6h.nc
"""

for request in requests.split('\n')[:]:
    command_line = 'wget {}'.format(request)
    print(command_line)
    subprocess.run(command_line, shell=True)