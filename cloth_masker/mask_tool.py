import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os

class ImageMaskingApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Masking Tool (Polygon + Zoom/Pan)")

        # --- Variables ---
        self.source_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.image_files = []
        self.current_image_index = -1
        self.current_pil_image = None    # Original PIL Image
        self.displayed_tk_image = None # PhotoImage object for canvas

        # Polygon drawing variables
        self.polygon_points_original = [] # Stores (x,y) tuples in ORIGINAL image coordinates
        self.polygon_canvas_item_ids = [] # Stores IDs of drawn lines/markers on canvas
        self.is_polygon_closed = False

        # Zoom and Pan variables
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        # pan_offset_orig: top-left corner of the viewport in ORIGINAL image coordinates
        self.pan_offset_orig_x = 0.0
        self.pan_offset_orig_y = 0.0

        # These are offsets for drawing the (potentially smaller than canvas) final image onto the canvas
        self.canvas_view_offset_x = 0 
        self.canvas_view_offset_y = 0
        # Actual width/height of the image portion being displayed on canvas
        self.displayed_image_width_on_canvas = 0
        self.displayed_image_height_on_canvas = 0


        # Panning state
        self.is_panning = False
        self.pan_start_mouse_x = 0
        self.pan_start_mouse_y = 0
        self.pan_start_offset_orig_x = 0
        self.pan_start_offset_orig_y = 0

        # --- UI Elements ---
        # Top frame for directory selection and zoom controls
        top_controls_frame = tk.Frame(master, padx=10, pady=5)
        top_controls_frame.pack(fill=tk.X)

        dir_frame = tk.Frame(top_controls_frame)
        dir_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Button(dir_frame, text="Select Source Dir", command=self.select_source_dir).pack(side=tk.LEFT, padx=5)
        self.source_label = tk.Label(dir_frame, text="No source selected", fg="gray", width=20, anchor='w')
        self.source_label.pack(side=tk.LEFT, padx=5)

        tk.Button(dir_frame, text="Select Output Dir", command=self.select_output_dir).pack(side=tk.LEFT, padx=(10,5))
        self.output_label = tk.Label(dir_frame, text="No output selected", fg="gray", width=20, anchor='w')
        self.output_label.pack(side=tk.LEFT, padx=5)

        zoom_frame = tk.Frame(top_controls_frame)
        zoom_frame.pack(side=tk.RIGHT)
        tk.Button(zoom_frame, text="Zoom In", command=lambda: self.apply_zoom(1.2)).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="Zoom Out", command=lambda: self.apply_zoom(1/1.2)).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=2)


        # Image Display Frame
        self.canvas_frame = tk.Frame(master, bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        
        self.canvas_width_default = 800 
        self.canvas_height_default = 600
        self.canvas = tk.Canvas(self.canvas_frame, bg="lightgray", 
                                width=self.canvas_width_default, height=self.canvas_height_default)
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # Event Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_left_mouse_down) # Polygon drawing
        self.canvas.bind("<ButtonPress-2>", self.on_middle_mouse_down) # Panning (Button-3 on macOS for middle)
        self.canvas.bind("<B2-Motion>", self.on_middle_mouse_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_mouse_up)
        # For macOS, middle mouse might be Button-3 if 2-button mouse + scroll wheel click
        # Or if trackpad is configured for 3-finger tap as middle click.
        # For broader compatibility, might need to check platform or offer alternative.
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel) # Windows & some Linux
        self.canvas.bind("<Button-4>", self.on_mouse_wheel) # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel) # Linux scroll down

        master.bind("<Configure>", self.on_window_resize)

        # Bottom Controls Frame (Polygon actions, Submit)
        bottom_controls_frame = tk.Frame(master, padx=10, pady=10)
        bottom_controls_frame.pack(fill=tk.X)

        self.status_label = tk.Label(bottom_controls_frame, text="Please select directories to start.")
        self.status_label.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        self.clear_polygon_button = tk.Button(bottom_controls_frame, text="Clear Polygon", command=self.clear_current_polygon, state=tk.DISABLED)
        self.clear_polygon_button.pack(side=tk.LEFT, padx=5)
        
        self.finish_polygon_button = tk.Button(bottom_controls_frame, text="Finish Polygon", command=self.finish_polygon, state=tk.DISABLED)
        self.finish_polygon_button.pack(side=tk.LEFT, padx=5)

        self.submit_button = tk.Button(bottom_controls_frame, text="Submit & Next", command=self.submit_mask, state=tk.DISABLED)
        self.submit_button.pack(side=tk.RIGHT, padx=5)

    # --- Coordinate Transformation ---
    def _canvas_to_original_coords(self, canvas_x, canvas_y):
        if not self.current_pil_image or self.displayed_image_width_on_canvas == 0: # or zoom_level is 0
            return None, None 
        
        # Point relative to the top-left of the displayed image portion on canvas
        img_coord_x = canvas_x - self.canvas_view_offset_x
        img_coord_y = canvas_y - self.canvas_view_offset_y
        
        # Convert to original image coordinates
        original_x = self.pan_offset_orig_x + (img_coord_x / self.zoom_level)
        original_y = self.pan_offset_orig_y + (img_coord_y / self.zoom_level)
        
        return original_x, original_y

    def _original_to_canvas_coords(self, original_x, original_y):
        if not self.current_pil_image:
            return None, None

        # Position relative to the pan offset, scaled by zoom
        img_coord_x = (original_x - self.pan_offset_orig_x) * self.zoom_level
        img_coord_y = (original_y - self.pan_offset_orig_y) * self.zoom_level
        
        # Add canvas view offset (where the image portion starts drawing on canvas)
        canvas_x = img_coord_x + self.canvas_view_offset_x
        canvas_y = img_coord_y + self.canvas_view_offset_y
        
        return canvas_x, canvas_y

    # --- Display Logic ---
    def _update_display(self):
        """High-level function to refresh the entire canvas view."""
        if self.current_pil_image:
            self._display_current_image_view()
            self._redraw_polygon_on_canvas() # Redraw polygon based on new view
        else:
            self.canvas.delete("all")
            self.displayed_tk_image = None
        self.check_button_states()

    def _display_current_image_view(self):
        """Renders the current view (zoomed/panned) of the original image."""
        if not self.current_pil_image:
            self.canvas.delete("all")
            return

        self.canvas.delete("all") # Clear previous drawings (image and polygon)
                                 # Polygon will be redrawn by _redraw_polygon_on_canvas

        orig_w, orig_h = self.current_pil_image.size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1: # Canvas not realized yet
            canvas_w = self.canvas_width_default
            canvas_h = self.canvas_height_default
        
        # Determine the portion of the original image to crop based on pan and zoom
        # This is the viewport in original image coordinates
        crop_x1 = self.pan_offset_orig_x
        crop_y1 = self.pan_offset_orig_y
        crop_w_orig = canvas_w / self.zoom_level # Width of original image portion to show
        crop_h_orig = canvas_h / self.zoom_level # Height of original image portion to show
        crop_x2 = crop_x1 + crop_w_orig
        crop_y2 = crop_y1 + crop_h_orig

        # Clamp crop box to actual image dimensions
        actual_crop_x1 = max(0, crop_x1)
        actual_crop_y1 = max(0, crop_y1)
        actual_crop_x2 = min(orig_w, crop_x2)
        actual_crop_y2 = min(orig_h, crop_y2)

        if actual_crop_x1 >= actual_crop_x2 or actual_crop_y1 >= actual_crop_y2:
            # Invalid crop region (e.g., panned too far out with high zoom)
            self.displayed_tk_image = None
            self.displayed_image_width_on_canvas = 0
            self.displayed_image_height_on_canvas = 0
            return

        pil_sub_image = self.current_pil_image.crop((int(actual_crop_x1), int(actual_crop_y1), 
                                                     int(actual_crop_x2), int(actual_crop_y2)))

        # Calculate the size this sub-image should be on canvas
        self.displayed_image_width_on_canvas = int(pil_sub_image.width * self.zoom_level)
        self.displayed_image_height_on_canvas = int(pil_sub_image.height * self.zoom_level)

        if self.displayed_image_width_on_canvas <=0 or self.displayed_image_height_on_canvas <=0:
            self.displayed_tk_image = None
            return

        resized_pil_sub_image = pil_sub_image.resize(
            (self.displayed_image_width_on_canvas, self.displayed_image_height_on_canvas), 
            Image.Resampling.LANCZOS
        )
        self.displayed_tk_image = ImageTk.PhotoImage(resized_pil_sub_image)

        # Calculate where to draw this resized sub-image on the canvas
        # This accounts for panning "off the edge" of the original image
        self.canvas_view_offset_x = 0
        if crop_x1 < 0: # Panned left, so image starts further right on canvas
            self.canvas_view_offset_x = int(-crop_x1 * self.zoom_level)
        
        self.canvas_view_offset_y = 0
        if crop_y1 < 0: # Panned up
            self.canvas_view_offset_y = int(-crop_y1 * self.zoom_level)
        
        # If the entire zoomed image is smaller than canvas, center it additionally
        # This case is mostly handled if pan_offset_orig is kept within bounds
        # such that the viewport always tries to fill the canvas.
        # However, if the *entire original image* at current zoom is smaller than canvas:
        total_zoomed_width = orig_w * self.zoom_level
        total_zoomed_height = orig_h * self.zoom_level

        if total_zoomed_width < canvas_w:
            self.canvas_view_offset_x = int((canvas_w - total_zoomed_width) / 2)
             # Adjust pan_offset_orig_x if we are centering the whole image
            if self.pan_offset_orig_x !=0: # if not already at 0,0 of original
                pass # This logic is tricky. For now, assume pan_offset is the primary driver.
                     # The reset_view will set pan_offset_orig to 0,0 for fit-to-screen.
        if total_zoomed_height < canvas_h:
            self.canvas_view_offset_y = int((canvas_h - total_zoomed_height) / 2)


        self.canvas.create_image(self.canvas_view_offset_x, self.canvas_view_offset_y,
                                 anchor=tk.NW, image=self.displayed_tk_image)

    # --- Zoom and Pan ---
    def apply_zoom(self, factor, zoom_center_canvas_x=None, zoom_center_canvas_y=None):
        if not self.current_pil_image:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if zoom_center_canvas_x is None:
            zoom_center_canvas_x = canvas_w / 2
        if zoom_center_canvas_y is None:
            zoom_center_canvas_y = canvas_h / 2
        
        # Point under cursor/center in original image space BEFORE zoom
        center_orig_x, center_orig_y = self._canvas_to_original_coords(zoom_center_canvas_x, zoom_center_canvas_y)
        if center_orig_x is None: return # Cannot determine original coords

        old_zoom = self.zoom_level
        self.zoom_level *= factor
        self.zoom_level = max(self.min_zoom, min(self.max_zoom, self.zoom_level)) # Clamp zoom

        if abs(self.zoom_level - old_zoom) < 0.001: # No significant change
            return

        # New pan offset so the original point (center_orig_x, center_orig_y)
        # remains at the same canvas position (zoom_center_canvas_x, zoom_center_canvas_y)
        # From _canvas_to_original_coords: orig_x = pan_x + (canvas_x - view_offset_x) / zoom
        # So, pan_x = orig_x - (canvas_x - view_offset_x) / zoom
        # Here, view_offset_x is the offset of the *displayed sub-image* on canvas.
        # For zooming at cursor, we want the point (center_orig_x, center_orig_y)
        # to map to (zoom_center_canvas_x, zoom_center_canvas_y) AFTER zoom.
        # The self.canvas_view_offset_x/y is determined by the pan and image edges.
        # Let's assume for this calculation that the image portion fills the canvas from (0,0)
        # i.e., canvas_view_offset_x/y are 0 for this specific calculation.
        # The actual display function will re-calculate canvas_view_offset_x/y correctly.
        self.pan_offset_orig_x = center_orig_x - (zoom_center_canvas_x / self.zoom_level)
        self.pan_offset_orig_y = center_orig_y - (zoom_center_canvas_y / self.zoom_level)
        
        self._ensure_pan_in_bounds()
        self._update_display()

    def _ensure_pan_in_bounds(self):
        """Adjusts pan_offset_orig_x/y to keep the image view reasonable."""
        if not self.current_pil_image: return

        orig_w, orig_h = self.current_pil_image.size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # Max pan: how far can the top-left of viewport (in orig coords) go
        # such that the image still touches the viewport's right/bottom edge.
        max_pan_x = orig_w - (canvas_w / self.zoom_level) if self.zoom_level > 0 else orig_w
        max_pan_y = orig_h - (canvas_h / self.zoom_level) if self.zoom_level > 0 else orig_h
        
        # Min pan is typically 0,0 but can be negative if image is smaller than canvas / zoom level
        # For now, let's keep it simple:
        min_pan_x = 0 
        min_pan_y = 0
        # A more sophisticated approach would allow panning such that the image edge can be anywhere on canvas.
        # For now, ensure some part of the image is visible if possible.
        # This current logic might be too restrictive if image is smaller than canvas.
        # Let's allow panning such that the image can be fully off-screen to one side,
        # but not so far that the *viewport* is entirely outside the image.
        
        # Effective width/height of the viewport in original image coordinates
        viewport_w_orig = canvas_w / self.zoom_level
        viewport_h_orig = canvas_h / self.zoom_level

        # Don't let pan_offset_orig_x be so large that viewport_x1 > orig_w
        self.pan_offset_orig_x = min(self.pan_offset_orig_x, orig_w - viewport_w_orig * 0.1) # allow 10% of viewport to be off
        self.pan_offset_orig_y = min(self.pan_offset_orig_y, orig_h - viewport_h_orig * 0.1)

        # Don't let pan_offset_orig_x be so small (negative) that viewport_x2 < 0
        self.pan_offset_orig_x = max(self.pan_offset_orig_x, -viewport_w_orig * 0.9) # allow 10% of viewport to be off
        self.pan_offset_orig_y = max(self.pan_offset_orig_y, -viewport_h_orig * 0.9)


    def reset_view(self):
        if not self.current_pil_image:
            return
        
        orig_w, orig_h = self.current_pil_image.size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <=1 or canvas_h <=1 : # Not realized
            canvas_w = self.canvas_width_default
            canvas_h = self.canvas_height_default


        if orig_w == 0 or orig_h == 0: return

        # Calculate zoom to fit entire image
        zoom_w = canvas_w / orig_w
        zoom_h = canvas_h / orig_h
        self.zoom_level = min(zoom_w, zoom_h)
        self.zoom_level = max(self.min_zoom, min(self.max_zoom, self.zoom_level))


        # Center the image: pan_offset_orig_x/y should be such that the image center
        # aligns with viewport center.
        # Or simpler for "fit" view: pan_offset_orig is (0,0) and display logic handles centering.
        self.pan_offset_orig_x = 0
        self.pan_offset_orig_y = 0
        
        self._update_display()

    # --- Event Handlers ---
    def on_window_resize(self, event=None):
        # For simplicity, reset view on resize. Could try to maintain zoom/pan.
        if self.current_pil_image:
            self.reset_view() 
            # If a polygon was being drawn, it's points are in original coords,
            # so _redraw_polygon_on_canvas will correctly place them.
            # However, if it was complex, user might prefer it cleared.
            # Current _update_display calls _redraw_polygon_on_canvas.

    def on_mouse_wheel(self, event):
        if not self.current_pil_image: return
        factor = 0
        if event.num == 5 or event.delta < 0: # Scroll down (zoom out)
            factor = 1 / 1.2
        if event.num == 4 or event.delta > 0: # Scroll up (zoom in)
            factor = 1.2
        
        if factor != 0:
            self.apply_zoom(factor, event.x, event.y)

    def on_left_mouse_down(self, event): # Polygon point
        if not self.current_pil_image: return

        if self.is_polygon_closed:
            self.clear_current_polygon() # Start new polygon if one was closed

        orig_x, orig_y = self._canvas_to_original_coords(event.x, event.y)
        if orig_x is None or orig_y is None: return

        # Optional: Check if click is within original image bounds (already implicitly handled by crop in display)
        # orig_w, orig_h = self.current_pil_image.size
        # if not (0 <= orig_x < orig_w and 0 <= orig_y < orig_h):
        #     return # Clicked outside image content area

        self.polygon_points_original.append((orig_x, orig_y))
        self._redraw_polygon_on_canvas() # Redraw with the new point
        self.check_button_states()

    def on_middle_mouse_down(self, event):
        if not self.current_pil_image: return
        self.is_panning = True
        self.pan_start_mouse_x = event.x
        self.pan_start_mouse_y = event.y
        self.pan_start_offset_orig_x = self.pan_offset_orig_x
        self.pan_start_offset_orig_y = self.pan_offset_orig_y
        self.canvas.config(cursor="fleur")

    def on_middle_mouse_drag(self, event):
        if not self.is_panning or not self.current_pil_image:
            return
        
        delta_canvas_x = event.x - self.pan_start_mouse_x
        delta_canvas_y = event.y - self.pan_start_mouse_y

        # Convert canvas delta to original image delta
        delta_orig_x = delta_canvas_x / self.zoom_level
        delta_orig_y = delta_canvas_y / self.zoom_level

        self.pan_offset_orig_x = self.pan_start_offset_orig_x - delta_orig_x # Drag right, pan left
        self.pan_offset_orig_y = self.pan_start_offset_orig_y - delta_orig_y
        
        self._ensure_pan_in_bounds()
        self._update_display()


    def on_middle_mouse_up(self, event):
        self.is_panning = False
        self.canvas.config(cursor="")

    # --- Directory and Image Loading ---
    def select_source_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.source_dir.set(dir_path)
            self.source_label.config(text=os.path.basename(dir_path), fg="black")
            self.load_image_list()
        self.check_button_states()

    def select_output_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir.set(dir_path)
            self.output_label.config(text=os.path.basename(dir_path), fg="black")
            if not os.path.exists(self.output_dir.get()):
                try:
                    os.makedirs(self.output_dir.get())
                except OSError as e:
                    messagebox.showerror("Error", f"Could not create output directory: {e}")
                    self.output_dir.set("")
                    self.output_label.config(text="No output selected", fg="gray")
        self.check_button_states()

    def load_image_list(self):
        s_dir = self.source_dir.get()
        # ... (rest of the function is largely the same as before)
        if not s_dir or not os.path.isdir(s_dir):
            self.image_files = []
            self.current_image_index = -1
            self.current_pil_image = None
            self._clear_polygon_state_full()
            self._update_display()
            self.status_label.config(text="Source directory is not valid.")
            return

        self.image_files = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
        for f_name in sorted(os.listdir(s_dir)):
            if f_name.lower().endswith(valid_extensions):
                self.image_files.append(os.path.join(s_dir, f_name))

        if self.image_files:
            self.current_image_index = 0
            self.load_current_image_file()
        else:
            self.current_image_index = -1
            self.current_pil_image = None
            self._clear_polygon_state_full()
            self._update_display()
            self.status_label.config(text="No images found in source directory.")
        self.check_button_states()


    def load_current_image_file(self): # Renamed from load_current_image
        if 0 <= self.current_image_index < len(self.image_files):
            image_path = self.image_files[self.current_image_index]
            try:
                img = Image.open(image_path)
                self.current_pil_image = img.convert("RGB") 
                self._clear_polygon_state_full() # Clear polygon for new image
                self.reset_view() # This calls _update_display
                self.status_label.config(text=f"Image {self.current_image_index + 1} of {len(self.image_files)}: {os.path.basename(image_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image {image_path}:\n{e}")
                self.current_pil_image = None
                self._clear_polygon_state_full()
                self._update_display() # Clear canvas
                # Try to load next image if current one fails
                self.current_image_index +=1
                if self.current_image_index < len(self.image_files):
                    self.load_current_image_file()
                else:
                    self.status_label.config(text="Error loading image. No more images.")
        else: # No more images or invalid index
            self.current_pil_image = None
            self._clear_polygon_state_full()
            self._update_display() # Clear canvas
            if self.image_files: 
                 self.status_label.config(text="All images processed!")
            elif not self.source_dir.get():
                 self.status_label.config(text="Please select directories to start.")
            else: # Source dir selected, but no images were loaded/found
                 self.status_label.config(text="No images loaded or finished processing.")
        self.check_button_states()


    # --- Polygon Logic ---
    def _redraw_polygon_on_canvas(self):
        # Clear only old polygon items, not the background image
        for item_id in self.polygon_canvas_item_ids:
            self.canvas.delete(item_id)
        self.polygon_canvas_item_ids = []

        if not self.polygon_points_original or not self.current_pil_image:
            return

        # Convert original points to current canvas view coordinates and draw
        canvas_points = []
        for ox, oy in self.polygon_points_original:
            cx, cy = self._original_to_canvas_coords(ox, oy)
            if cx is not None: # If conversion is valid
                canvas_points.append((cx, cy))
        
        if not canvas_points: return

        # Draw lines and markers
        for i, (cx, cy) in enumerate(canvas_points):
            r = 3
            marker_id = self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill="red", outline="red")
            self.polygon_canvas_item_ids.append(marker_id)
            if i > 0:
                p1 = canvas_points[i-1]
                p2 = canvas_points[i]
                line_id = self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="red", width=2)
                self.polygon_canvas_item_ids.append(line_id)
        
        # If polygon is closed, draw closing line
        if self.is_polygon_closed and len(canvas_points) >= 3:
            p_last = canvas_points[-1]
            p_first = canvas_points[0]
            line_id = self.canvas.create_line(p_last[0], p_last[1], p_first[0], p_first[1], 
                                              fill="red", width=2, dash=(4, 2))
            self.polygon_canvas_item_ids.append(line_id)

    def _clear_polygon_state_full(self):
        """Clears polygon points and visual items, resets closed state."""
        for item_id in self.polygon_canvas_item_ids:
            self.canvas.delete(item_id)
        self.polygon_points_original = []
        self.polygon_canvas_item_ids = []
        self.is_polygon_closed = False
        # self._update_display() # Not always needed here, often called by parent
        self.check_button_states()

    def clear_current_polygon(self):
        """User action to clear the current polygon drawing."""
        self._clear_polygon_state_full()
        self._redraw_polygon_on_canvas() # Essential to remove visuals if any were missed
                                         # and to reflect empty state.
        # self._update_display() # This would redraw the image too, might be overkill.
                               # _redraw_polygon_on_canvas should suffice.

    def finish_polygon(self):
        if not self.current_pil_image or len(self.polygon_points_original) < 3:
            messagebox.showwarning("Polygon Incomplete", "A polygon must have at least 3 points.")
            return
        if self.is_polygon_closed:
            messagebox.showinfo("Polygon Info", "Polygon is already closed.")
            return
        self.is_polygon_closed = True
        self._redraw_polygon_on_canvas() # Redraw to show the closing line
        self.check_button_states()

    def submit_mask(self):
        if not self.current_pil_image or not self.output_dir.get() or \
           not self.is_polygon_closed or len(self.polygon_points_original) < 3:
            messagebox.showwarning("Cannot Submit", "Ensure image, closed polygon (>=3 pts), and output dir are set.")
            return

        try:
            orig_w, orig_h = self.current_pil_image.size
            mask_image = Image.new("L", (orig_w, orig_h), "black")
            draw = ImageDraw.Draw(mask_image)
            
            # Convert original_points to integer tuples for PIL draw
            pil_polygon_points = [(int(round(x)), int(round(y))) for x, y in self.polygon_points_original]
            draw.polygon(pil_polygon_points, fill="white", outline=None)

            original_filename = os.path.basename(self.image_files[self.current_image_index])
            mask_save_path = os.path.join(self.output_dir.get(), original_filename)
            if not mask_save_path.lower().endswith(".png"):
                 mask_save_path = os.path.splitext(mask_save_path)[0] + ".png"
            mask_image.save(mask_save_path, "PNG")
            
            self.current_image_index += 1
            self.load_current_image_file() # This will reset polygon state and view

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create or save mask: {e}")
        self.check_button_states() # Ensure states are correct

    # --- UI State ---
    def check_button_states(self):
        # Submit button
        can_submit = (self.current_pil_image and self.output_dir.get() and
                      self.is_polygon_closed and len(self.polygon_points_original) >= 3)
        self.submit_button.config(state=tk.NORMAL if can_submit else tk.DISABLED)

        # Finish Polygon button
        can_finish = (self.current_pil_image and not self.is_polygon_closed and
                      len(self.polygon_points_original) >= 3)
        self.finish_polygon_button.config(state=tk.NORMAL if can_finish else tk.DISABLED)
            
        # Clear Polygon button
        can_clear = self.current_pil_image and len(self.polygon_points_original) > 0
        self.clear_polygon_button.config(state=tk.NORMAL if can_clear else tk.DISABLED)

    def run(self):
        # Call reset_view once after window is mapped to get initial canvas size
        self.master.after(100, self.reset_view_if_no_image)
        self.master.mainloop()

    def reset_view_if_no_image(self):
        # This is to ensure canvas is sized before first reset_view if no image is loaded initially
        if not self.current_pil_image:
            # If there's no image, reset_view won't do much.
            # The canvas should be blank or show a placeholder.
            # The main thing is that canvas dimensions are known for when an image IS loaded.
            pass 


if __name__ == '__main__':
    root = tk.Tk()
    root.minsize(950, 750) # Slightly larger default minsize
    app = ImageMaskingApp(root)
    app.run()
