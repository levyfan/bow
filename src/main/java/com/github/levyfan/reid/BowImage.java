package com.github.levyfan.reid;

import com.github.levyfan.reid.bow.Strip;
import com.github.levyfan.reid.bow.StripMethod;
import com.github.levyfan.reid.sp.SuperPixel;
import com.github.levyfan.reid.sp.SuperPixelMethond;

import java.awt.image.BufferedImage;

/**
 * @author fanliwen
 */
public class BowImage {

    public BufferedImage image;

    public BufferedImage image4;

    public BufferedImage mask;

    public BufferedImage mask4;

    public SuperPixel[] sp4;

    public Strip[] strip4;

    public BowImage(SuperPixelMethond spMethod, StripMethod stripMethod, BufferedImage image, BufferedImage mask) {
        this.image = image;
        this.image4 = ImageUtils.resize(image, 4, false);
        this.sp4 = spMethod.generate(image4);

        if (mask != null) {
            this.mask = mask;
            this.mask4 = ImageUtils.resize(mask, 4, true);
            this.strip4 = stripMethod.strip(sp4, mask4);
        }
    }
}
