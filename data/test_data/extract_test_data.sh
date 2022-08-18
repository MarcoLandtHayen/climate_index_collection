
source_dir="/gxfs_work1/geomar/smomw235/data_user/data_mlandt/"

# maybe create targe dirs
mkdir -p FOCI CESM

# make sure we have NCO in the path
module load nco

# extract 120 time steps of CESM data
ncks -O -d time,1,120,1 \
	/gxfs_work1/geomar/smomw235/data_user/data_mlandt/CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.cam2.h0.0001-0999.PSL.midmonth.nc \
	CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.cam2.h0.0001-0010.PSL.nc
ncks -O -d time,1,120,1 \
	/gxfs_work1/geomar/smomw235/data_user/data_mlandt/CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.cam2.h0.0001-0999.SST.midmonth.nc \
	CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.cam2.h0.0001-0010.SST.nc
ncks -O -d time,1,120,1 \
	/gxfs_work1/geomar/smomw235/data_user/data_mlandt/CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.cam2.h0.0001-0999.Z500.midmonth.nc \
	CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.cam2.h0.0001-0010.Z500.nc
ncks -O -d time,1,120,1 \
        /gxfs_work1/geomar/smomw235/data_user/data_mlandt/CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.cam2.h0.0001-0999.TS.nc \
        CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.cam2.h0.0001-0010.TS.nc
ncks -O -d time,1,120,1 -v SALT \
        /gxfs_work1/geomar/smomw235/data_user/data_mlandt/CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.pop.h.0001-0999.SSS.nc \
        CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.pop.h.0001-0010.SSS.nc
ncks -O -d time,1,120,1 \
        /gxfs_work1/geomar/smomw235/data_user/data_mlandt/CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.cam2.h0.0001-0999.PRECT.nc \
        CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.cam2.h0.0001-0010.PRECT.nc


# extract 120 time steps of FOCI data

ncks -O -d time,1,120,1 \
	/gxfs_work1/geomar/smomw235/data_user/data_mlandt/FOCI/FOCI1.3-SW038_echam6_ATM_mm_2350-3349_geopoth_pl_monthly_50000_midmonth.nc \
	FOCI/FOCI1.3-SW038_echam6_ATM_mm_2350-2359_geopoth_pl_monthly_50000.nc
ncks -O -d time,1,120,1 \
	/gxfs_work1/geomar/smomw235/data_user/data_mlandt/FOCI/FOCI1.3-SW038_echam6_BOT_mm_2350-3349_slp_monthly_1_midmonth.nc \
	FOCI/FOCI1.3-SW038_echam6_BOT_mm_2350-2359_slp_monthly_1.nc
ncks -O -d time,1,120,1 \
	/gxfs_work1/geomar/smomw235/data_user/data_mlandt/FOCI/FOCI1.3-SW038_echam6_BOT_mm_2350-3349_tsw_monthly_1_midmonth.nc \
	FOCI/FOCI1.3-SW038_echam6_BOT_mm_2350-2359_tsw_monthly_1.nc
ncks -O -d time,1,120,1 \
        /gxfs_work1/geomar/smomw235/data_user/data_mlandt/FOCI/FOCI1.3-SW038_echam6_BOT_mm_2350-3349_temp2_monthly_1.nc \
        FOCI/FOCI1.3-SW038_echam6_BOT_mm_2350-2359_temp2_monthly_1.nc
ncks -O -d time_counter,1,120,1 -v sosaline \
        /gxfs_work1/geomar/smomw235/data_user/data_mlandt/FOCI/FOCI1.3-SW038_1m_23500101_33491231_grid_T.nc \
        FOCI/FOCI1.3-SW038_1m_23500101_23591231_grid_T.nc
ncks -O -d time,1,120,1 \
        /gxfs_work1/geomar/smomw235/data_user/data_mlandt/FOCI/FOCI1.3-SW038_echam6_BOT_mm_2350-3349_precip_monthly_1_midmonth.nc \
        FOCI/FOCI1.3-SW038_echam6_BOT_mm_2350-2359_precip_monthly_1.nc
