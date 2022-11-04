-- Flagging issues in data quality/cleaning to resolve

-- Checking column types of IDs across the schema
select * from information_schema.columns
where table_schema = 'cleaned'
  and column_name ilike '%id%';
 
-- Building Info Table

-- Parcels that don't merge with Monthly Tax records
select bi.*
from cleaned.building_info bi left join cleaned.monthly_tax mt on upper(bi.parcelid) = mt.parcelid
where mt.parcelid is null
order by bi.parcelid;

-- Duplicates in Parcel ID for relevant property classes
with relevant as (
select bi.*, mt.prop_class_code, mt.class_description
from cleaned.building_info bi join cleaned.monthly_tax mt using(parcelid)
where prop_class_code in (510, 550, 520, 401, 530, 625, 404, 470)
)
select * 
from relevant
where parcelid in (select parcelid from relevant group by parcelid having count(*) > 1)
order by parcelid;

-- Zero values in fields that shouldn't be zero

select bi.*, mt.prop_class_code, mt.class_description
from cleaned.building_info bi join cleaned.monthly_tax mt using(parcelid)
where prop_class_code in (510, 550, 520, 401, 530, 625, 404, 470)
and ("LIVE_FSQFT" = 0);

select bi.*, mt.prop_class_code, mt.class_description
from cleaned.building_info bi join cleaned.monthly_tax mt using(parcelid)
where prop_class_code in (510, 550, 520, 401, 530, 625, 404, 470)
and ("LIVE_FSQFT" > 0) and "STORYHT" = 0;

-- Mismatch in stories and sqft reporting (# of floors)
select bi.*, mt.prop_class_code, mt.class_description
from cleaned.building_info bi join cleaned.monthly_tax mt using(parcelid)
where prop_class_code in (510, 550, 520, 401, 530, 625, 404, 470)
and ("SQFT_FLR2" > 0) and "STORYHT" <= 1;

-- Historical Sales

-- Mismatched use code info between historical sales and monthly tax
select hs.parcelid, hs.use_code, mt.prop_class_code
from cleaned.historic_sales hs left join cleaned.monthly_tax mt using(parcelid)
where use_code != prop_class_code;

-- Num stories/total finish area/etc. = 0
select parcelid, use_code, number_of_parcels, sale_price, date_of_sale, tax_year, basement, 
       heating, air_conditioning, total_rooms, full_bath, half_bath, fireplaces, 
       garage_type, garage_capacity, num_stories, year_built, finished_sq_ft, total_finish_area, 
       first_floor_area, half_floor_area, finished_basement 
from cleaned.historic_sales hs where num_stories = 0
and use_code in (510, 550, 520, 401, 530, 625, 404, 470);

-- Zero sale price observations
select parcelid, use_code, instrument_type, number_of_parcels, sale_price, date_of_sale, tax_year, 
       basement, heating, air_conditioning, total_rooms, full_bath, half_bath, fireplaces, 
       garage_type, garage_capacity, num_stories, year_built, finished_sq_ft, total_finish_area, 
       first_floor_area, half_floor_area, finished_basement  
from cleaned.historic_sales hs where sale_price = 0
and use_code in (510, 550, 520, 401, 530, 625, 404, 470);

-- Monthly Tax

-- 0 deeded acreage cases
select parcelid, prop_class_code, class_description, property_description, deeded_acreage, sale_amount, 
       num_parcels, market_land_value, market_impr_value, total_market_value, transfer_date, conveyance_number, 
       deed_number, divided_flag, new_construction_flag, spec_assess_flag, foreclosure_flag, rental_registration_flag
from cleaned.monthly_tax mt
where prop_class_code in (510, 550, 520, 401, 530, 625, 404, 470)
and deeded_acreage = 0;

-- What property classes are the 0 deeded acreage cases concentrated in?
select prop_class_code, count(*)
from cleaned.monthly_tax mt
where prop_class_code in (510, 550, 520, 401, 530, 625, 404, 470)
and deeded_acreage = 0
group by prop_class_code;

-- 0 sale price cases
select parcelid, prop_class_code, owner_name_1, owner_name_2, previous_owner_name_1, 
       previous_owner_name_2, class_description, property_description, deeded_acreage, sale_amount, 
       num_parcels, market_land_value, market_impr_value, total_market_value, transfer_date, conveyance_number, 
       deed_number, divided_flag, new_construction_flag, spec_assess_flag, foreclosure_flag, rental_registration_flag
from cleaned.monthly_tax mt
where prop_class_code in (510, 550, 520, 401, 530, 625, 404, 470)
and sale_amount = 0;

-- 0 total market value
select parcelid, prop_class_code, class_description, property_description, deeded_acreage, sale_amount, 
       num_parcels, market_land_value, market_impr_value, total_market_value, transfer_date, conveyance_number, 
       deed_number, divided_flag, new_construction_flag, spec_assess_flag, foreclosure_flag, rental_registration_flag
from cleaned.monthly_tax mt
where prop_class_code in (510, 550, 520, 401, 530, 625, 404, 470)
and total_market_value = 0;

-- Property transfers

-- not merging to monthly tax
select count(distinct pt.parcelid)
from cleaned.property_transfer pt left join cleaned.monthly_tax mt on upper(pt.parcelid) = mt.parcelid
where mt.parcelid is null;

select extract(year from date_of_sale) "year_of_sale", count(distinct pt.parcelid)
from cleaned.property_transfer pt left join cleaned.monthly_tax mt on upper(pt.parcelid) = mt.parcelid
where mt.parcelid is null
group by extract(year from date_of_sale);

select num_parcels_sold, count(distinct pt.parcelid)
from cleaned.property_transfer pt left join cleaned.monthly_tax mt on upper(pt.parcelid) = mt.parcelid
where mt.parcelid is null
group by num_parcels_sold;



select pt.parcelid, num_parcels_sold, date_of_sale, property_class, building_value, land_value, sale_price, valid_sale, deed_type, conveyance_no, is_new
from cleaned.property_transfer pt left join cleaned.monthly_tax mt on upper(pt.parcelid) = mt.parcelid
where mt.parcelid is null
order by pt.parcelid;

-- Duplicates - is conveyance_no a good candidate for uid?
select count(*) from cleaned.property_transfer pt where conveyance_no is null;
select count(distinct conveyance_no) from cleaned.property_transfer pt where conveyance_no is not null;

-- Parcels that have duplicate sales records
select count(distinct pt.parcelid)
from cleaned.property_transfer pt 
where (parcelid, date_of_sale) in (select parcelid, date_of_sale from cleaned.property_transfer pt2 group by parcelid, date_of_sale having count(*) > 1)
and property_class != '   '
and property_class::INT in (510, 550, 520, 401, 530, 625, 404, 470);

select pt.parcelid, num_parcels_sold, date_of_sale, property_class, building_value, land_value, sale_price, valid_sale, deed_type, conveyance_no, is_new
from cleaned.property_transfer pt 
where (parcelid, date_of_sale) in (select parcelid, date_of_sale from cleaned.property_transfer pt2 group by parcelid, date_of_sale having count(*) > 1)
and property_class != '   '
and property_class::INT in (510, 550, 520, 401, 530, 625, 404, 470)
order by parcelid, date_of_sale;

-- Building/land value = 0
select pt.parcelid, num_parcels_sold, date_of_sale, property_class, building_value, land_value, sale_price, valid_sale, deed_type, conveyance_no, is_new
from cleaned.property_transfer pt 
where building_value = 0 and land_value = 0
and property_class != '   '
and property_class::INT in (510, 550, 520, 401, 530, 625, 404, 470)
order by parcelid, date_of_sale;



