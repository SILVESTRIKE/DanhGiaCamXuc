ASPECTS = {
    "Nhà hàng": [
        'FOOD#QUALITY', 'FOOD#PRICES', 'SERVICE#GENERAL', 'AMBIENCE#GENERAL',
        'LOCATION#GENERAL', 'RESTAURANT#GENERAL', 'FOOD#STYLE&OPTIONS',
        'RESTAURANT#PRICES', 'RESTAURANT#MISCELLANEOUS', 'DRINKS#QUALITY',
        'DRINKS#PRICES', 'DRINKS#STYLE&OPTIONS'
    ],
    "Khách sạn": [
        'HOTEL#GENERAL', 'HOTEL#COMFORT', 'HOTEL#CLEANLINESS', 'HOTEL#DESIGN&FEATURES',
        'HOTEL#QUALITY', 'HOTEL#PRICES', 'ROOMS#GENERAL', 'ROOMS#CLEANLINESS',
        'ROOMS#DESIGN&FEATURES', 'ROOMS#COMFORT', 'ROOMS#QUALITY', 'ROOMS#PRICES',
        'ROOM_AMENITIES#GENERAL', 'ROOM_AMENITIES#CLEANLINESS', 'ROOM_AMENITIES#QUALITY',
        'ROOM_AMENITIES#DESIGN&FEATURES', 'ROOM_AMENITIES#COMFORT', 'FACILITIES#GENERAL',
        'FACILITIES#QUALITY', 'FACILITIES#DESIGN&FEATURES', 'FACILITIES#CLEANLINESS',
        'FACILITIES#COMFORT', 'FACILITIES#PRICES', 'LOCATION#GENERAL', 'SERVICE#GENERAL',
        'FOOD&DRINKS#QUALITY', 'FOOD&DRINKS#STYLE&OPTIONS', 'FOOD&DRINKS#PRICES',
        'FOOD&DRINKS#MISCELLANEOUS'
    ]
}

ASPECT_KEYWORDS = {
    "Nhà hàng": {
        'FOOD#QUALITY': ['ngon', 'dở', 'tươi', 'chất lượng', 'vị', 'thức ăn', 'đồ ăn'],
        'FOOD#PRICES': ['giá', 'rẻ', 'đắt', 'tiền', 'chi phí'],
        'SERVICE#GENERAL': ['phục vụ', 'nhân viên', 'nhanh', 'chậm', 'thái độ'],
        'AMBIENCE#GENERAL': ['không gian', 'view', 'sạch', 'bẩn', 'thoáng', 'đẹp'],
        'LOCATION#GENERAL': ['vị trí', 'địa điểm', 'dễ tìm', 'xa', 'gần'],
        'RESTAURANT#GENERAL': ['quán', 'nhà hàng', 'tổng thể', 'trải nghiệm'],
        'FOOD#STYLE&OPTIONS': ['món', 'menu', 'đa dạng', 'lựa chọn', 'kiểu'],
        'RESTAURANT#PRICES': ['giá cả', 'chi phí', 'hợp lý', 'đắt đỏ'],
        'RESTAURANT#MISCELLANEOUS': ['gửi xe', 'vệ sinh', 'khăn lạnh', 'tiện ích'],
        'DRINKS#QUALITY': ['nước uống', 'trà', 'nước ngọt', 'đồ uống'],
        'DRINKS#PRICES': ['giá', 'rẻ', 'đắt', 'nước uống'],
        'DRINKS#STYLE&OPTIONS': ['nước uống', 'menu', 'đa dạng']
    },
    "Khách sạn": {
        'HOTEL#GENERAL': ['khách sạn', 'tổng thể', 'trải nghiệm'],
        'HOTEL#COMFORT': ['thoải mái', 'yên tĩnh', 'ồn ào'],
        'HOTEL#CLEANLINESS': ['sạch sẽ', 'bẩn', 'vệ sinh'],
        'HOTEL#DESIGN&FEATURES': ['đẹp', 'thiết kế', 'kiến trúc', 'mới'],
        'HOTEL#QUALITY': ['chất lượng', 'tiêu chuẩn'],
        'HOTEL#PRICES': ['giá', 'rẻ', 'đắt', 'hợp lý'],
        'ROOMS#GENERAL': ['phòng', 'phòng ốc'],
        'ROOMS#CLEANLINESS': ['sạch', 'bẩn', 'vệ sinh phòng'],
        'ROOMS#DESIGN&FEATURES': ['rộng', 'chật', 'thiết kế phòng'],
        'ROOMS#COMFORT': ['thoải mái', 'khó chịu', 'giường'],
        'ROOMS#QUALITY': ['chất lượng phòng'],
        'ROOMS#PRICES': ['giá phòng', 'chi phí'],
        'ROOM_AMENITIES#GENERAL': ['tiện nghi', 'đồ dùng'],
        'ROOM_AMENITIES#CLEANLINESS': ['không sạch', 'không bẩn'],
        'ROOM_AMENITIES#QUALITY': ['chất lượng đồ dùng'],
        'ROOM_AMENITIES#DESIGN&FEATURES': ['thiết kế đồ dùng'],
        'ROOM_AMENITIES#COMFORT': ['thoải mái đồ dùng'],
        'FACILITIES#GENERAL': ['cơ sở', 'dịch vụ'],
        'FACILITIES#QUALITY': ['chất lượng cơ sở'],
        'FACILITIES#DESIGN&FEATURES': ['thiết kế cơ sở', 'bể bơi'],
        'FACILITIES#CLEANLINESS': ['sạch cơ sở'],
        'FACILITIES#COMFORT': ['thoải mái cơ sở'],
        'FACILITIES#PRICES': ['giá dịch vụ'],
        'LOCATION#GENERAL': ['vị trí', 'địa điểm', 'gần', 'xa'],
        'SERVICE#GENERAL': ['phục vụ', 'nhân viên', 'thái độ'],
        'FOOD&DRINKS#QUALITY': ['ngon', 'dở', 'chất lượng đồ ăn'],
        'FOOD&DRINKS#STYLE&OPTIONS': ['đa dạng', 'menu', 'lựa chọn'],
        'FOOD&DRINKS#PRICES': ['giá đồ ăn', 'rẻ', 'đắt'],
        'FOOD&DRINKS#MISCELLANEOUS': ['bữa sáng', 'buffet']
    }
}

LABEL_ENCODER = ['neutral','positive','negative', ]
