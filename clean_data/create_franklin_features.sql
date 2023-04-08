-- Create Franklin County feature data

drop table if exists processed.franklin_features;

-- creating tables only choosing relevant prop class codes
-- Join only keeps the buildings that merge into tax info
create table processed.franklin_features as
(select d."PARCEL ID" as parcelid,
		case when d."ATTIC" in ('0 - NO ATTIC', '1 - ATTIC UNF') then 'NO ATTIC'
			 when d."ATTIC" = '4 - FULL ATTIC FINISH' then 'FULL ATTIC'
		     else 'PARTIAL ATTIC'
		     end as attic_cat,
		d."FINBSMTAREA" as bsmt_sqft,
		d."SFLA" as live_sqft,
		d."FLR1AREA" as sqft_flr1,
		substring(d."STORIES" FROM '^[\d\.]+')::double precision as story_ht,
		d."YRBLT" as year_built,
		pt."LUC" as prop_class_code,
		pt."LastSaleParcelCount" as number_of_parcels,
		d."STYLE" as style,
		case when d."GRADE" in ('E-1 - POOR -', 'E+2 - POOR ++', 'E+1 - POOR +', 'E - POOR QUALITY') then 1
			 when d."GRADE" in ('D-2 - POOR ++', 'D-1 - BELOW AVERAGE -') then 2
			 when d."GRADE" in ('D+2 - BELOW AVERAGE ++', 'D+1 - BELOW AVERAGE +', 'D - BELOW AVERAGE QUALITY') then 3
			 when d."GRADE" in ('C-2 - AVERAGE - -') then 4
			 when d."GRADE" in ('C-1 - AVERAGE -', 'C - AVERAGE QUALITY', 'C+1 - AVERAGE +') then 5
			 when d."GRADE" in ('C+2 - AVERAGE + +', 'B-1 - GOOD -', 'B - GOOD QUALITY', 'B+1 - GOOD +') then 6
			 when d."GRADE" in ('B+2 - GOOD ++') then 7
			 when d."GRADE" in ('A-1 - VERY GOOD -', 'A - VERY GOOD QUALITY', 'A+1 - VERY GOOD +') then 8
			 else 9 end as grade_numeric,
		case when d."EXTWALL" = '2 - STUCCO' then 'Stucco'
			 when d."EXTWALL" = '3 - TILE' then 'Tile'
			 when d."EXTWALL" in ('91 - 1/6 MASONRY TO FRAME .167', '92 - 2/6 MASONRY TO FRAME .333') then 'F/M 91-92'
			 when d."EXTWALL" in ('93 - 3/6 MASONRY TO FRAME .500', '94 - 4/6 MASONRY TO FRAME .667') then 'F/M 93-94'
			 when d."EXTWALL" = '95 - 5/6 MASONRY TO FRAME .833' then 'F/M 95'
			 when d."EXTWALL" = '8 - STONE' then 'Stone'
			 when d."EXTWALL" = '7 - BRICK' then 'Brick'
			 when d."EXTWALL" = '6 - CONCRETE' then 'Concrete Block'
			 when d."EXTWALL" = '4 - CONCRETE BLOCK' then 'Concrete'
			 when d."EXTWALL" = '5 - METAL' then 'Metal'
			 when d."EXTWALL" = '1 - WD/ALUM/VINYL' then 'Frame/Siding'
			 else null end as exterior_wall_type,
		case when d."BSMT" ilike '%%CRAWL%%' then 'Crawl Space'
			 when d."BSMT" ilike '%%BASEMENT%%' then 'Basement'
			 else null end as basement_grouped,
		case when d."HEAT" = '0 - NONE' then 'None' else 'Base' end as heating,
		case when d."HEAT" = '2 - HEAT/AIR CON' then 'Central' else 'None' end as air_conditioning,
		d."RMTOT" as total_rooms,
		d."FIXBATH" as full_bath,
		d."FIXHALF" as half_bath,
		d."WBFP_S" as fireplaces,
		substring(d."CARSPACESATT" FROM '^[\d\.]+')::double precision as garage_capacity
from raw_franklin."Dwelling" d join raw_franklin."Parcel_Tax" pt on d."PARCEL ID" = pt."PARCEL ID" -- parcels that match with tax data
where pt."LUC" in (510, 550, 520, 401, 530, 625, 404, 470) -- use code falls in correct types
);
