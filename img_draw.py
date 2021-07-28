from PIL import Image, ImageDraw
import itertools
import functools

def table_offsets(draw, playerA, playerB, resultrows):
    margin = 2

    a_width = draw.textlength(playerA) + 5
    b_width = draw.textlength(playerB) + 5

    awins_width = functools.reduce(max, map(lambda x: draw.textlength(x[0]), resultrows)) + 5
    bwins_width = functools.reduce(max, map(lambda x: draw.textlength(x[1]), resultrows)) + 5
    list_width  = functools.reduce(max, map(lambda x: draw.textlength(x[2]), resultrows)) + 5

    return tuple(itertools.accumulate((margin, a_width, awins_width, bwins_width, b_width, list_width)))

def impl_image(playerA, playerB, resultrows):
    im = Image.new("RGB", (512,256), color='white')
    draw = ImageDraw.Draw(im)
    impl_colors = ['red', 'green', 'blue']

    rows = []
    for groups, color in zip(itertools.groupby(resultrows, lambda x: set(x[2])), impl_colors):
        rows.extend((*elem, color) for elem in groups[1])

    # Get the table-internal offsets
    offsets = table_offsets(draw, playerA, playerB, rows)

    # Center the table on the image
    tablestep = 16
    tablesize = (offsets[-1], tablestep*len(resultrows))
    tablepos = ((512-tablesize[0])/2, 30)

    # Make a title
    title = f'{playerA} vs. {playerB}'
    titlepos = ((512 - draw.textlength(title))/2, 10)

    # Create image
    draw.text(titlepos, title, fill=0)
    for row,ypos in zip(rows, itertools.count(tablepos[1], tablestep)):
        draw.line((tablepos[0], ypos, tablepos[0]+offsets[5]+offsets[0], ypos), fill=0)
        draw.text((tablepos[0]+offsets[0], ypos+2), playerA, fill=0)
        draw.text((tablepos[0]+offsets[1], ypos+2), row[0], fill=0)
        draw.text((tablepos[0]+offsets[2], ypos+2), row[1], fill=0)
        draw.text((tablepos[0]+offsets[3], ypos+2), playerB, fill=0)
        draw.rectangle((tablepos[0]+offsets[4]-offsets[0], ypos+1, tablepos[0]+offsets[5]+offsets[0], ypos+tablestep-1), fill = row[3])
        draw.text((tablepos[0]+offsets[4], ypos+2), row[2], fill=0)
    draw.line((tablepos[0], tablepos[1]+offsets[5]+offsets[0], tablepos[0]+tablesize[0], tablepos[1]+offsets[5]+offsets[0]), fill=0)

    im.show()

