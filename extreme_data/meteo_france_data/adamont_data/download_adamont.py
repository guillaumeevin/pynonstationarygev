import subprocess

"""
Go to: https://drias-prod.meteo.fr/
On the top left, select "simulation au format netcdf" instead of "simulation au format csv"
Then, select the fact that you want to url
"""

requests = """https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/CNRM-CM5/ALADIN53/rcp8.5/day/snow/Snow_FORCING_CNRM-ALADIN53_CNRM-CERFACS-CNRM-CM5_RCP85_alp_2005080106_2100080106_daysum.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/IPSL-CM5A/WRF331F/rcp8.5/day/snow/Snow_FORCING_IPSL-INERIS-WRF331F_IPSL-IPSL-CM5A-MR_RCP85_alp_2005080106_2100080106_daysum.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/CNRM-CM5/RCA4/rcp8.5/day/snow/Snow_FORCING_SMHI-RCA4_CNRM-CERFACS-CNRM-CM5_RCP85_alp_2005080106_2100080106_daysum.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/EC-EARTH/RCA4/rcp8.5/day/snow/Snow_FORCING_SMHI-RCA4_ICHEC-EC-EARTH_RCP85_alp_2005080106_2100080106_daysum.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MOHC-HadGEM2/RCA4/rcp8.5/day/snow/Snow_FORCING_SMHI-RCA4_MOHC-HadGEM2-ES_RCP85_alp_2005080106_2099080106_daysum.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/IPSL-CM5A/RCA4/rcp8.5/day/snow/Snow_FORCING_SMHI-RCA4_IPSL-IPSL-CM5A-MR_RCP85_alp_2005080106_2100080106_daysum.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MPI-ESM-LR/RCA4/rcp8.5/day/snow/Snow_FORCING_SMHI-RCA4_MPI-M-MPI-ESM-LR_RCP85_alp_2005080106_2100080106_daysum.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MOHC-HadGEM2/RACMO22E/rcp8.5/day/snow/Snow_FORCING_KNMI-RACMO22E_MOHC-HadGEM2-ES_RCP85_alp_2005080106_2099080106_daysum.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/CNRM-CM5/CCLM4-8-17/rcp8.5/day/snow/Snow_FORCING_CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5_RCP85_alp_2005080106_2100080106_daysum.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/EC-EARTH/CCLM4-8-17/rcp8.5/day/snow/Snow_FORCING_CLMcom-CCLM4-8-17_ICHEC-EC-EARTH_RCP85_alp_2005080106_2100080106_daysum.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MOHC-HadGEM2/CCLM4-8-17/rcp8.5/day/snow/Snow_FORCING_CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES_RCP85_alp_2005080106_2099080106_daysum.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MPI-ESM-LR/CCLM4-8-17/rcp8.5/day/snow/Snow_FORCING_CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR_RCP85_alp_2005080106_2100080106_daysum.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MPI-ESM-LR/REMO019/rcp8.5/day/snow/Snow_FORCING_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_RCP85_alp_2005080106_2100080106_daysum.nc
https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/NorESM1/HIRHAM5/rcp8.5/day/snow/Snow_FORCING_DMI-HIRHAM5_NCC-NorESM1-M_RCP85_alp_2005080106_2100080106_daysum.nc
"""

# requests = """https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/CNRM-CM5/ALADIN53/historical/day/snowswe/SNOWSWE_PRO_CNRM-ALADIN53_CNRM-CERFACS-CNRM-CM5_HISTO_alp_1950080106_2005073106_6h.nc
# https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/IPSL-CM5A/WRF331F/historical/day/snowswe/SNOWSWE_PRO_IPSL-INERIS-WRF331F_IPSL-IPSL-CM5A-MR_HISTO_alp_1951080106_2005073106_6h.nc
# https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/CNRM-CM5/RCA4/historical/day/snowswe/SNOWSWE_PRO_SMHI-RCA4_CNRM-CERFACS-CNRM-CM5_HISTO_alp_1970080106_2005073106_6h.nc
# https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/EC-EARTH/RCA4/historical/day/snowswe/SNOWSWE_PRO_SMHI-RCA4_ICHEC-EC-EARTH_HISTO_alp_1970080106_2005073106_6h.nc
# https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MOHC-HadGEM2/RCA4/historical/day/snowswe/SNOWSWE_PRO_SMHI-RCA4_MOHC-HadGEM2-ES_HISTO_alp_1981080106_2005073106_6h.nc
# https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/IPSL-CM5A/RCA4/historical/day/snowswe/SNOWSWE_PRO_SMHI-RCA4_IPSL-IPSL-CM5A-MR_HISTO_alp_1970080106_2005073106_6h.nc
# https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MPI-ESM-LR/RCA4/historical/day/snowswe/SNOWSWE_PRO_SMHI-RCA4_MPI-M-MPI-ESM-LR_HISTO_alp_1970080106_2005073106_6h.nc
# https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MOHC-HadGEM2/RACMO22E/historical/day/snowswe/SNOWSWE_PRO_KNMI-RACMO22E_MOHC-HadGEM2-ES_HISTO_alp_1981080106_2005073106_6h.nc
# https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/CNRM-CM5/CCLM4-8-17/historical/day/snowswe/SNOWSWE_PRO_CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5_HISTO_alp_1950080106_2005073106_6h.nc
# https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/EC-EARTH/CCLM4-8-17/historical/day/snowswe/SNOWSWE_PRO_CLMcom-CCLM4-8-17_ICHEC-EC-EARTH_HISTO_alp_1950080106_2005073106_6h.nc
# https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MOHC-HadGEM2/CCLM4-8-17/historical/day/snowswe/SNOWSWE_PRO_CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES_HISTO_alp_1981080106_2005073106_6h.nc
# https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MPI-ESM-LR/CCLM4-8-17/historical/day/snowswe/SNOWSWE_PRO_CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR_HISTO_alp_1950080106_2005073106_6h.nc
# https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/MPI-ESM-LR/REMO019/historical/day/snowswe/SNOWSWE_PRO_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_HISTO_alp_1950080106_2005073106_6h.nc
# https://climatedata.umr-cnrm.fr/public/dcsc/projects/DRIAS/ADAMONT2017/Alpes/NorESM1/HIRHAM5/historical/day/snowswe/SNOWSWE_PRO_DMI-HIRHAM5_NCC-NorESM1-M_HISTO_alp_1951080106_2005073106_6h.nc
# """

for request in requests.split('\n')[:]:
    command_line = 'wget {}'.format(request)
    print(command_line)
    subprocess.run(command_line, shell=True)