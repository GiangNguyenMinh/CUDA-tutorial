/*
 * Được thực thi trên 'cudart' library
 * Được link qua static library: libcudart.a hoặc qua dynamic library: libcudart.
 * Các API của runime library được bắt đầu vơi tiền tố 'cu'
 */

/* -------------------------------------------------------------------------------------------------------------------- */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!! Initialization !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/*
* Không có một hàm khởi tạo cụ thể cho runtime, nó được khởi tạo đầu tiên khi một hàm runtime được gọi
 * Runtime tạo ra một CUDA context cho mỗi device system. Context là 'primary context' cho thiết bị đó và được khởi tạo
 * * trong hàm rutime đầu tiên, cái yêu cầu một hoạt động của context trên device đó. Nó được chia sẻ giữa tất cả các
 * * host threads của ứng dụng. Một phần của việc tạo ra context, device code được thực thi just-in-time nếu cần thiết
 * * và được load vào device memory.
 *
 *  - cuCtxGetCurrent(): truy suất context được tạo ra trong quá trình khởi tạo
 *  - CUdeviceptr: tạo ra pointer (!!sử dụng driver API) --> ex: CUdeviceptr ptr; (CUdeviceptr)normalptr
 *  - cudaDeviceReset(): sẽ xóa 'primary context' của device cái mà host thread đang hoạt động trên, và bất cứ thao tác
 *  gọi runtime function tiếp theo bởi host thread sẽ tạo ra 'primary context' mới cho device.
 *  - cudaSetDevice(): chọn device.
 */

/*
 *  (*) Trong CUDA 12.0, cudaSetDevice() bây giờ sẽ khởi tạo rõ runtime sau khi thay đổi thiết bị hiện tại cho host
 *  (*) * thread (vì thế nên check cudaSetDevice() để xem khởi tạo runtime có lỗi không)
 *  (*) Trong những phiên bản trước thì khởi tạo runtime sẽ bị delay trên new device cho đến khi lệnh runtime đầu tiên
 *  (*) * được gọi sau cudaSetDevice().
 */