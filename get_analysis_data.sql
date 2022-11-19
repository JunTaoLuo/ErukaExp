WITH relevant_bi AS (
        SELECT *
        FROM processed.building_info
        WHERE parcelid IN
            (SELECT parcelid
            FROM processed.building_info
            GROUP BY parcelid
            HAVING (sum(case when year_built < 1930 then 1 else 0 end) = 1) and (count(*) = 1) -- parcels that only have one building, and that building was built < 1930
            )
            AND story_ht != 0 -- parcels without 0 story_ht (removing half finished or unbuilt buildings)
    ),
    relevant_hs AS (
        SELECT *
        FROM processed.historic_sales
        WHERE year_built < 1930 and year_built is not null
    ),
    bi_hs_match AS (
        SELECT parcelid, rb.year_built, rb.attic_sqft, rb.bsmt_sqft, rb.live_sqft, rb.sqft_flr1, rb.sqft_flr2, rb.sqft_flrh, rb.story_ht, rb.prop_class_code, rb.class_description, 
               rh.number_of_parcels, rh.date_of_sale, rh.appraisal_area, rh.total_sales_records, rh.sale_price, 
               rh.style, rh.grade, rh.exterior_wall_type, rh.basement, rh.heating, rh.air_conditioning, rh.total_rooms, rh.full_bath, rh.half_bath, rh.fireplaces, rh.garage_type, rh.garage_capacity,
               rh.finished_sq_ft, rh.total_finish_area, rh.first_floor_area, rh.half_floor_area, rh.finished_basement
        FROM relevant_bi rb JOIN relevant_hs rh USING(parcelid) -- only keep parcels that match between datasets
        WHERE rb.year_built = rh.year_built -- only keep those with consistent year_buitl information, so we know all columns relevant for that building
    ),
    labels AS (
        SELECT parcel_number as "parcelid", initial_land_value, initial_building_value -- get hand-labeled data
        FROM ocrtargets
    )
    SELECT *
    FROM bi_hs_match LEFT JOIN labels USING(parcelid); -- merge the land-labeled data onto relevant parcels
